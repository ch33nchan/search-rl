import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple
import re


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
        max_length: int = 512
    ):
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        )
        self.model.eval()
    
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
        for doc in documents:
            score = self.score_single(query, doc)
            scores.append(score)
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

