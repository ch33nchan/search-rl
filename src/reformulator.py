import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional


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
        max_new_tokens: int = 128
    ):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        )
        self.model.eval()
    
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

