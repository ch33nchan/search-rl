import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple
import re
from .utils import get_device, get_dtype, optimize_for_metal


JUDGE_PROMPT = """You are a search relevance judge. Rate how relevant the document is to the query on a scale of 0-10.

Examples:
Query: "machine learning optimization algorithms"
Document: "Stochastic gradient descent (SGD) is an iterative method for optimizing an objective function with suitable smoothness properties. It can be regarded as a stochastic approximation of gradient descent optimization."
Score: 9

Query: "best pizza recipes"
Document: "The history of pizza begins in antiquity, as various ancient cultures produced basic flatbreads with several toppings."
Score: 3

Query: "python async programming"
Document: "Asyncio is a library to write concurrent code using the async/await syntax. It is used as a foundation for multiple Python asynchronous frameworks."
Score: 10

Query: "climate change effects on polar bears"
Document: "The stock market showed significant volatility this quarter with major indices fluctuating widely."
Score: 0

Now rate:
Query: "{query}"
Document: "{document}"

Respond with only a number 0-10."""


class RewardModel:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "cuda",
        max_length: int = 512,
        use_quantization: bool = False,
        gpu_id: int = 0,  # Specific GPU to use
        batch_size: int = 8
    ):
        self.device = get_device(device)
        self.max_length = max_length
        self.gpu_id = gpu_id
        self.batch_size = batch_size
        dtype = get_dtype(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        load_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True
        }
        
        if self.device == "mps":
            load_kwargs["device_map"] = None
        elif self.device == "cuda":
            # Check if GPU ID is valid
            if gpu_id >= torch.cuda.device_count():
                raise ValueError(
                    f"Invalid gpu_id {gpu_id}. Only {torch.cuda.device_count()} GPUs are visible. "
                    "If using CUDA_VISIBLE_DEVICES, remember that indices are remapped to 0, 1, etc."
                )
            # Use specific GPU
            load_kwargs["device_map"] = f"cuda:{gpu_id}"
        else:
            load_kwargs["device_map"] = self.device
        
        if use_quantization and self.device != "mps":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype
            )
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        
        if self.device == "mps":
            self.model = self.model.to(self.device)
            self.model = optimize_for_metal(self.model, self.device)
        
        self.model.eval()
        
        # Update device string for tensor operations
        if self.device == "cuda":
            self.device = f"cuda:{gpu_id}"
    
    def _truncate_text(self, text: str, max_chars: int = 500) -> str:
        if len(text) > max_chars:
            return text[:max_chars] + "..."
        return text
    
    def _parse_score(self, response: str) -> float:
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', response)
        if numbers:
            score = float(numbers[0])
            return min(max(score, 0.0), 10.0)
        return 5.0
    
    @torch.no_grad()
    def score_single(self, query: str, document: str) -> float:
        doc_text = self._truncate_text(
            f"{document.get('title', '')} {document.get('text', '')}"
            if isinstance(document, dict) else document
        )
        
        prompt = JUDGE_PROMPT.format(query=query, document=doc_text)
        
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return self._parse_score(response)
    
    def score_batch(self, query: str, documents: List[Dict]) -> np.ndarray:
        scores = []
        prompts = []
        
        for doc in documents:
            doc_text = self._truncate_text(
                f"{doc.get('title', '')} {doc.get('text', '')}"
                if isinstance(doc, dict) else doc
            )
            prompt = JUDGE_PROMPT.format(query=query, document=doc_text)
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(text)
            
        # Process in batches
        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i:i + self.batch_size]
            
            inputs = self.tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=self.max_length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode responses
            input_len = inputs['input_ids'].shape[1]
            for output in outputs:
                response = self.tokenizer.decode(output[input_len:], skip_special_tokens=True)
                scores.append(self._parse_score(response))
                
        return np.array(scores)
    
    def compute_ndcg(
        self,
        query: str,
        documents: List[Dict],
        k: int = 10
    ) -> Tuple[float, np.ndarray]:
        documents = documents[:k]
        scores = self.score_batch(query, documents)
        
        dcg = sum(scores[i] / np.log2(i + 2) for i in range(len(scores)))
        
        sorted_scores = np.sort(scores)[::-1]
        ideal_dcg = sum(sorted_scores[i] / np.log2(i + 2) for i in range(len(sorted_scores)))
        
        if ideal_dcg == 0:
            ndcg = 0.0
        else:
            ndcg = dcg / ideal_dcg
        
        return float(np.clip(ndcg, 0.0, 1.0)), scores
    
    def compute_mrr(self, query: str, documents: List[Dict], threshold: float = 7.0) -> float:
        scores = self.score_batch(query, documents)
        
        for i, score in enumerate(scores):
            if score >= threshold:
                return 1.0 / (i + 1)
        return 0.0

