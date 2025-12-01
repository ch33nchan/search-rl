import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    doc_ids: List[str]
    scores: np.ndarray
    documents: List[Dict]
    query_embedding: np.ndarray


class Retriever:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda"
    ):
        self.device = device
        self.encoder = SentenceTransformer(model_name, device=device)
        self.index: Optional[faiss.Index] = None
        self.doc_ids: List[str] = []
        self.documents: Dict[str, Dict] = {}
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
    
    def build_index(self, corpus: Dict[str, Dict]):
        self.documents = corpus
        self.doc_ids = list(corpus.keys())
        
        texts = [
            f"{doc.get('title', '')} {doc.get('text', '')}"
            for doc in corpus.values()
        ]
        
        embeddings = self.encoder.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings.astype(np.float32))
    
    def save_index(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        np.save(path / "doc_ids.npy", np.array(self.doc_ids))
    
    def load_index(self, path: Path, corpus: Dict[str, Dict]):
        self.index = faiss.read_index(str(path / "index.faiss"))
        self.doc_ids = np.load(path / "doc_ids.npy").tolist()
        self.documents = corpus
    
    def encode_query(self, query: str) -> np.ndarray:
        embedding = self.encoder.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding
    
    def search(self, query: str, top_k: int = 10) -> RetrievalResult:
        query_embedding = self.encode_query(query)
        
        scores, indices = self.index.search(
            query_embedding.reshape(1, -1).astype(np.float32),
            top_k
        )
        
        retrieved_ids = [self.doc_ids[i] for i in indices[0]]
        retrieved_docs = [self.documents[doc_id] for doc_id in retrieved_ids]
        
        return RetrievalResult(
            doc_ids=retrieved_ids,
            scores=scores[0],
            documents=retrieved_docs,
            query_embedding=query_embedding
        )
    
    def search_with_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> Tuple[List[str], np.ndarray, List[Dict]]:
        scores, indices = self.index.search(
            query_embedding.reshape(1, -1).astype(np.float32),
            top_k
        )
        
        retrieved_ids = [self.doc_ids[i] for i in indices[0]]
        retrieved_docs = [self.documents[doc_id] for doc_id in retrieved_ids]
        
        return retrieved_ids, scores[0], retrieved_docs
    
    def get_aggregated_embedding(self, doc_ids: List[str]) -> np.ndarray:
        texts = [
            f"{self.documents[did].get('title', '')} {self.documents[did].get('text', '')}"
            for did in doc_ids
        ]
        embeddings = self.encoder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings.mean(axis=0)

