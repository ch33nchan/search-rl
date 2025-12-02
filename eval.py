import argparse
import torch
from pathlib import Path

from src.config import TrainConfig
from src.retriever import Retriever
from src.reward import RewardModel
from src.reformulator import Reformulator
from src.policy import PolicyNetwork
from src.environment import SearchEnv
from src.evaluation import Evaluator
from src.data import (
    load_beir_dataset,
    beir_queries_to_synthetic,
    split_queries,
    load_query_bank
)


def load_policy(checkpoint_path: Path, config: TrainConfig, device: str) -> PolicyNetwork:
    policy = PolicyNetwork(
        embedding_dim=config.policy.embedding_dim,
        hidden_dim=config.policy.hidden_dim,
        action_dim=config.policy.action_dim,
        gru_layers=config.policy.gru_layers,
        dropout=config.policy.dropout
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    
    return policy


def evaluate(
    checkpoint_path: Path,
    dataset_name: str,
    device: str,
    num_queries: int = 200
):
    config = TrainConfig(dataset_name=dataset_name, device=device)
    config.optimize_for_device()
    
    corpus, queries, qrels = load_beir_dataset(dataset_name)
    
    query_bank_path = Path("data") / dataset_name / "query_bank.json"
    if query_bank_path.exists():
        all_queries = load_query_bank(query_bank_path)
    else:
        all_queries = beir_queries_to_synthetic(queries, qrels)
    
    _, test_queries = split_queries(all_queries, train_ratio=0.8, seed=config.seed)
    test_queries = test_queries[:num_queries]
    
    retriever = Retriever(
        model_name=config.retriever.model_name,
        device=device
    )
    
    index_path = Path("data") / dataset_name / "index"
    if index_path.exists():
        retriever.load_index(index_path, corpus)
    else:
        retriever.build_index(corpus)
        retriever.save_index(index_path)
    
    reward_model = RewardModel(
        model_name=config.reward.model_name,
        device=device,
        max_length=config.reward.max_length
    )
    
    reformulator = Reformulator(
        model_name=config.reformulator.model_name,
        device=device,
        max_new_tokens=config.reformulator.max_new_tokens
    )
    
    env = SearchEnv(
        retriever=retriever,
        reward_model=reward_model,
        reformulator=reformulator,
        query_bank=test_queries,
        max_steps=config.env.max_steps,
        top_k=config.retriever.top_k,
        device=device
    )
    
    policy = load_policy(checkpoint_path, config, device)
    
    evaluator = Evaluator(env, policy, device)
    
    results = evaluator.compare_all(test_queries, verbose=True)
    evaluator.print_comparison(results)
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="nfcorpus")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-queries", type=int, default=200)
    args = parser.parse_args()
    
    evaluate(
        checkpoint_path=Path(args.checkpoint),
        dataset_name=args.dataset,
        device=args.device,
        num_queries=args.num_queries
    )


if __name__ == "__main__":
    main()

