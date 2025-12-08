import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional
from .utils import get_device, get_dtype, optimize_for_metal


NARROW_PROMPT = """Given a search query and the top retrieved results, rewrite the query to be more specific and focused.

Current query: {query}

Top results:
{results}

Write a more specific version of this query that would retrieve more relevant documents. Only output the new query, nothing else."""


BROAD_PROMPT = """Given a search query and the top retrieved results, rewrite the query to be broader and less restrictive.

Current query: {query}

Top results:
{results}

Write a more general version of this query that would retrieve more diverse documents. Only output the new query, nothing else."""


class Reformulator:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "cuda",
        max_new_tokens: int = 128,
        use_quantization: bool = False,
        gpu_id: int = 1  # Specific GPU to use (default to second GPU)
    ):
        self.device = get_device(device)
        self.max_new_tokens = max_new_tokens
        self.gpu_id = gpu_id
        dtype = get_dtype(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='left')
        
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
    
    def _format_results(self, documents: List[Dict], max_docs: int = 3) -> str:
        result_strs = []
        for i, doc in enumerate(documents[:max_docs]):
            title = doc.get('title', '')
            text = doc.get('text', '')[:200]
            result_strs.append(f"{i+1}. {title}: {text}...")
        return "\n".join(result_strs)
    
    @torch.no_grad()
    def reformulate(
        self,
        query: str,
        documents: List[Dict],
        mode: str = "narrow"
    ) -> str:
        results_text = self._format_results(documents)
        
        if mode == "narrow":
            prompt = NARROW_PROMPT.format(query=query, results=results_text)
        else:
            prompt = BROAD_PROMPT.format(query=query, results=results_text)
        
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        new_query = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        new_query = new_query.split('\n')[0].strip()
        if len(new_query) < 3 or len(new_query) > 500:
            return query
        
        return new_query
    
    def narrow(self, query: str, documents: List[Dict]) -> str:
        return self.reformulate(query, documents, mode="narrow")
    
    def broaden(self, query: str, documents: List[Dict]) -> str:
        return self.reformulate(query, documents, mode="broad")

