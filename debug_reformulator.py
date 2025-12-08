import torch
from src.config import TrainConfig
from src.retriever import Retriever
from src.reformulator import Reformulator
from src.data import load_beir_dataset, beir_queries_to_synthetic
import random

def main():
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = "fiqa"
    
    print(f"Loading {dataset} dataset...")
    corpus, queries, qrels = load_beir_dataset(dataset)
    synthetic_queries = beir_queries_to_synthetic(queries, qrels)
    
    print("Initializing models...")
    retriever = Retriever(device=device)
    retriever.build_index(corpus)
    
    reformulator = Reformulator(
        device=device,
        gpu_id=0  # Use first GPU for debugging
    )
    
    # Pick 5 random queries
    test_queries = random.sample(synthetic_queries, 5)
    
    print("\n=== Reformulator Inspection ===\n")
    
    for q in test_queries:
        print(f"Original Query: '{q.query}'")
        
        # 1. Initial Search
        results = retriever.search(q.query, top_k=3)
        print(f"Top Result: {results.documents[0]['title']}")
        
        # 2. Narrow
        narrowed = reformulator.reformulate(q.query, results.documents, mode="narrow")
        print(f"Narrowed:       '{narrowed}'")
        
        # 3. Broaden
        broadened = reformulator.reformulate(q.query, results.documents, mode="broad")
        print(f"Broadened:      '{broadened}'")
        
        print("-" * 50)

if __name__ == "__main__":
    main()
