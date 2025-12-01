import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from beir import util
from beir.datasets.data_loader import GenericDataLoader
import numpy as np

from .query_gen import SyntheticQuery


BEIR_DATASETS = {
    "nfcorpus": "nfcorpus",
    "scifact": "scifact",
    "fiqa": "fiqa-2018",
    "arguana": "arguana",
    "scidocs": "scidocs"
}


def load_beir_dataset(
    dataset_name: str,
    data_dir: Path = Path("data")
) -> Tuple[Dict[str, Dict], Dict[str, Dict], Dict[str, Dict[str, int]]]:
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if dataset_name in BEIR_DATASETS:
        beir_name = BEIR_DATASETS[dataset_name]
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{beir_name}.zip"
        data_path = util.download_and_unzip(url, str(data_dir))
    else:
        data_path = data_dir / dataset_name
    
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    
    return corpus, queries, qrels


def beir_queries_to_synthetic(
    queries: Dict[str, Dict],
    qrels: Dict[str, Dict[str, int]],
    relevance_threshold: int = 1
) -> List[SyntheticQuery]:
    synthetic_queries = []
    
    for qid, query_text in queries.items():
        if qid not in qrels:
            continue
        
        relevant_docs = [
            doc_id for doc_id, rel in qrels[qid].items()
            if rel >= relevance_threshold
        ]
        
        if relevant_docs:
            synthetic_queries.append(SyntheticQuery(
                query=query_text,
                relevant_doc_ids=relevant_docs,
                query_type="beir"
            ))
    
    return synthetic_queries


def split_queries(
    queries: List[SyntheticQuery],
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[SyntheticQuery], List[SyntheticQuery]]:
    np.random.seed(seed)
    indices = np.random.permutation(len(queries))
    
    split_idx = int(len(queries) * train_ratio)
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    train_queries = [queries[i] for i in train_idx]
    test_queries = [queries[i] for i in test_idx]
    
    return train_queries, test_queries


def save_query_bank(queries: List[SyntheticQuery], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "query": q.query,
            "relevant_doc_ids": q.relevant_doc_ids,
            "query_type": q.query_type
        }
        for q in queries
    ]
    with open(path, "w") as f:
        json.dump(data, f)


def load_query_bank(path: Path) -> List[SyntheticQuery]:
    with open(path) as f:
        data = json.load(f)
    return [
        SyntheticQuery(
            query=d["query"],
            relevant_doc_ids=d["relevant_doc_ids"],
            query_type=d["query_type"]
        )
        for d in data
    ]


def save_corpus_index(
    corpus: Dict[str, Dict],
    doc_ids: List[str],
    path: Path
):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"corpus": corpus, "doc_ids": doc_ids}, f)


def load_corpus_index(path: Path) -> Tuple[Dict[str, Dict], List[str]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["corpus"], data["doc_ids"]

