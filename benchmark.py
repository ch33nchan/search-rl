import time
import torch
from src.utils import get_device
from src.retriever import Retriever
from src.reward import RewardModel
from src.reformulator import Reformulator
from src.policy import PolicyNetwork
from src.data import load_beir_dataset


def benchmark_device(device: str = None):
    device = get_device(device)
    print(f"\n{'='*60}")
    print(f"Benchmarking on device: {device}")
    print(f"{'='*60}\n")
    
    corpus, queries, qrels = load_beir_dataset("nfcorpus")
    sample_docs = list(corpus.values())[:10]
    sample_query = list(queries.values())[0] if queries else "test query"
    
    print("1. Retriever initialization...")
    start = time.time()
    retriever = Retriever(device=device)
    retriever.build_index({k: corpus[k] for k in list(corpus.keys())[:100]})
    retriever_time = time.time() - start
    print(f"   Time: {retriever_time:.2f}s")
    
    print("\n2. Reward model initialization...")
    start = time.time()
    reward_model = RewardModel(
        model_name="Qwen/Qwen2.5-1.5B-Instruct" if device == "mps" else "Qwen/Qwen2.5-3B-Instruct",
        device=device
    )
    reward_init_time = time.time() - start
    print(f"   Time: {reward_init_time:.2f}s")
    
    print("\n3. Reward scoring (10 docs)...")
    start = time.time()
    for doc in sample_docs[:10]:
        _ = reward_model.score_single(sample_query, doc)
    reward_score_time = time.time() - start
    print(f"   Time: {reward_score_time:.2f}s ({reward_score_time/10:.2f}s per doc)")
    
    print("\n4. Reformulator initialization...")
    start = time.time()
    reformulator = Reformulator(
        model_name="Qwen/Qwen2.5-1.5B-Instruct" if device == "mps" else "Qwen/Qwen2.5-3B-Instruct",
        device=device
    )
    reform_init_time = time.time() - start
    print(f"   Time: {reform_init_time:.2f}s")
    
    print("\n5. Query reformulation...")
    start = time.time()
    _ = reformulator.narrow(sample_query, sample_docs[:3])
    reform_time = time.time() - start
    print(f"   Time: {reform_time:.2f}s")
    
    print("\n6. Policy forward pass...")
    policy = PolicyNetwork().to(device)
    import numpy as np
    query_emb = np.random.randn(1, 5, 384).astype(np.float32)
    action_ids = np.random.randint(0, 4, (1, 5))
    result_emb = np.random.randn(1, 5, 384).astype(np.float32)
    
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = policy(
                torch.tensor(query_emb).to(device),
                torch.tensor(action_ids).to(device),
                torch.tensor(result_emb).to(device)
            )
    policy_time = time.time() - start
    print(f"   Time: {policy_time:.2f}s ({policy_time/100*1000:.2f}ms per forward)")
    
    print(f"\n{'='*60}")
    print("Estimated Training Time (10k episodes):")
    print(f"{'='*60}")
    
    episode_time = (
        retriever_time / 100 +
        reward_score_time / 10 * 2 +
        reform_time * 2 +
        policy_time / 100 * 5
    )
    
    total_time_hours = (episode_time * 10000) / 3600
    
    print(f"Per episode: ~{episode_time:.2f}s")
    print(f"10k episodes: ~{total_time_hours:.1f} hours")
    print(f"With 5 steps/episode avg: ~{total_time_hours * 1.5:.1f} hours")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    
    benchmark_device(args.device)

