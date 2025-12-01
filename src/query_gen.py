import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import random
from tqdm import tqdm


SINGLE_DOC_PROMPT = """Given this document, generate {n} different search queries that someone might type to find this document. The queries should be natural and diverse.

Document title: {title}
Document text: {text}

Generate exactly {n} queries, one per line. Only output the queries, nothing else."""


MULTI_DOC_PROMPT = """Given these related documents, generate a search query that would require information from multiple of them to answer well.

Documents:
{documents}

Generate one complex query that relates to multiple documents. Only output the query, nothing else."""


ADVERSARIAL_PROMPT = """Given this document, generate a search query where obvious keyword matches would lead to wrong documents, but semantic understanding would find this document.

Document title: {title}
Document text: {text}

Generate one query that requires semantic understanding. Only output the query, nothing else."""


@dataclass
class SyntheticQuery:
    query: str
    relevant_doc_ids: List[str]
    query_type: str


class QueryGenerator:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "cuda",
        queries_per_doc: int = 3
    ):
        self.device = device
        self.queries_per_doc = queries_per_doc
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
            trust_remote_code=True
        )
        self.model.eval()
    
    def _truncate_text(self, text: str, max_chars: int = 800) -> str:
        if len(text) > max_chars:
            return text[:max_chars] + "..."
        return text
    
    @torch.no_grad()
    def _generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        return self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
    
    def generate_single_doc_queries(
        self,
        doc_id: str,
        document: Dict
    ) -> List[SyntheticQuery]:
        title = document.get('title', '')
        text = self._truncate_text(document.get('text', ''))
        
        prompt = SINGLE_DOC_PROMPT.format(
            n=self.queries_per_doc,
            title=title,
            text=text
        )
        
        response = self._generate(prompt)
        queries = [q.strip() for q in response.split('\n') if q.strip()]
        
        return [
            SyntheticQuery(
                query=q,
                relevant_doc_ids=[doc_id],
                query_type="single"
            )
            for q in queries[:self.queries_per_doc]
            if len(q) > 5
        ]
    
    def generate_multi_doc_query(
        self,
        doc_ids: List[str],
        documents: List[Dict]
    ) -> Optional[SyntheticQuery]:
        docs_text = "\n\n".join([
            f"Document {i+1}:\nTitle: {doc.get('title', '')}\nText: {self._truncate_text(doc.get('text', ''), 400)}"
            for i, doc in enumerate(documents)
        ])
        
        prompt = MULTI_DOC_PROMPT.format(documents=docs_text)
        query = self._generate(prompt, max_new_tokens=100)
        query = query.split('\n')[0].strip()
        
        if len(query) > 5:
            return SyntheticQuery(
                query=query,
                relevant_doc_ids=doc_ids,
                query_type="multi"
            )
        return None
    
    def generate_adversarial_query(
        self,
        doc_id: str,
        document: Dict
    ) -> Optional[SyntheticQuery]:
        title = document.get('title', '')
        text = self._truncate_text(document.get('text', ''))
        
        prompt = ADVERSARIAL_PROMPT.format(title=title, text=text)
        query = self._generate(prompt, max_new_tokens=100)
        query = query.split('\n')[0].strip()
        
        if len(query) > 5:
            return SyntheticQuery(
                query=query,
                relevant_doc_ids=[doc_id],
                query_type="adversarial"
            )
        return None
    
    def generate_query_bank(
        self,
        corpus: Dict[str, Dict],
        num_single: int = 1000,
        num_multi: int = 200,
        num_adversarial: int = 200
    ) -> List[SyntheticQuery]:
        queries = []
        doc_ids = list(corpus.keys())
        
        sampled_single = random.sample(doc_ids, min(num_single // self.queries_per_doc, len(doc_ids)))
        for doc_id in tqdm(sampled_single, desc="Generating single-doc queries"):
            qs = self.generate_single_doc_queries(doc_id, corpus[doc_id])
            queries.extend(qs)
        
        for _ in tqdm(range(num_multi), desc="Generating multi-doc queries"):
            sample_ids = random.sample(doc_ids, min(3, len(doc_ids)))
            sample_docs = [corpus[did] for did in sample_ids]
            q = self.generate_multi_doc_query(sample_ids, sample_docs)
            if q:
                queries.append(q)
        
        sampled_adv = random.sample(doc_ids, min(num_adversarial, len(doc_ids)))
        for doc_id in tqdm(sampled_adv, desc="Generating adversarial queries"):
            q = self.generate_adversarial_query(doc_id, corpus[doc_id])
            if q:
                queries.append(q)
        
        return queries

