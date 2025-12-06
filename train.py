import argparse
import torch
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config import TrainConfig
from src.retriever import Retriever
from src.reward import RewardModel
from src.reformulator import Reformulator
from src.policy import PolicyNetwork, PolicyState
from src.environment import SearchEnv
from src.ppo import PPOTrainer, Transition
from src.evaluation import Evaluator
from src.data import (
    load_beir_dataset,
    beir_queries_to_synthetic,
    split_queries,
    save_query_bank,
    load_query_bank
)
from src.query_gen import QueryGenerator


def setup_components(config: TrainConfig, corpus, train_queries):
    retriever = Retriever(
        model_name=config.retriever.model_name,
        device=config.device
    )
    
    index_path = Path("data") / config.dataset_name / "index"
    if index_path.exists():
        retriever.load_index(index_path, corpus)
    else:
        retriever.build_index(corpus)
        retriever.save_index(index_path)
    
    reward_model = RewardModel(
        model_name=config.reward.model_name,
        device=config.device,
        max_length=config.reward.max_length,
        use_quantization=config.reward.use_quantization
    )
    
    reformulator = Reformulator(
        model_name=config.reformulator.model_name,
        device=config.device,
        max_new_tokens=config.reformulator.max_new_tokens,
        use_quantization=config.reformulator.use_quantization
    )
    
    env = SearchEnv(
        retriever=retriever,
        reward_model=reward_model,
        reformulator=reformulator,
        query_bank=train_queries,
        max_steps=config.env.max_steps,
        top_k=config.retriever.top_k,
        device=config.device,
        step_penalty=config.env.step_penalty
    )
    
    policy = PolicyNetwork(
        embedding_dim=config.policy.embedding_dim,
        hidden_dim=config.policy.hidden_dim,
        action_dim=config.policy.action_dim,
        gru_layers=config.policy.gru_layers,
        dropout=config.policy.dropout
    ).to(config.device)
    
    return retriever, reward_model, reformulator, env, policy


def train(config: TrainConfig):
    config.optimize_for_device()
    
    print(f"Using device: {config.device}")
    if config.device == "mps":
        print("Metal Performance Shaders (MPS) enabled")
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    config.log_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(config.log_dir)
    
    corpus, queries, qrels = load_beir_dataset(config.dataset_name)
    
    query_bank_path = Path("data") / config.dataset_name / "query_bank.json"
    
    if query_bank_path.exists():
        all_queries = load_query_bank(query_bank_path)
    else:
        all_queries = beir_queries_to_synthetic(queries, qrels)
        
        if len(all_queries) < 500:
            query_gen = QueryGenerator(
                model_name=config.query_gen.model_name,
                device=config.device,
                queries_per_doc=config.query_gen.queries_per_doc
            )
            synthetic = query_gen.generate_query_bank(
                corpus,
                num_single=500,
                num_multi=100,
                num_adversarial=100
            )
            all_queries.extend(synthetic)
        
        save_query_bank(all_queries, query_bank_path)
    
    train_queries, test_queries = split_queries(all_queries, train_ratio=0.8, seed=config.seed)
    
    retriever, reward_model, reformulator, env, policy = setup_components(
        config, corpus, train_queries
    )
    
    ppo_trainer = PPOTrainer(
        policy=policy,
        lr=config.ppo.lr,
        gamma=config.ppo.gamma,
        gae_lambda=config.ppo.gae_lambda,
        clip_epsilon=config.ppo.clip_epsilon,
        value_coef=config.ppo.value_coef,
        entropy_coef=config.ppo.entropy_coef,
        max_grad_norm=config.ppo.max_grad_norm,
        epochs_per_update=config.ppo.epochs_per_update,
        batch_size=config.ppo.batch_size,
        device=config.device
    )
    
    evaluator = Evaluator(env, policy, config.device)
    
    global_step = 0
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # Track action distribution during training
    
    for episode in tqdm(range(config.total_episodes), desc="Training"):
        obs, info = env.reset()
        policy_state = PolicyState(config.policy.hidden_dim, config.device)
        
        episode_reward = 0
        episode_steps = 0
        done = False
        
        while not done:
            state = policy_state.get_state()
            action, log_prob, value = policy.get_action(state, deterministic=False)
            action_counts[action] += 1  # Track actions
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            policy_state.update(
                policy.trajectory_encoder,
                obs["query_embedding"],
                action,
                obs["result_embedding"]
            )
            
            transition = Transition(
                state=state.squeeze(0),
                action=action,
                log_prob=log_prob.item(),
                value=value.item(),
                reward=reward,  # Include step penalty in all rewards
                done=done
            )
            ppo_trainer.store_transition(transition)
            
            episode_reward += reward  # Accumulate all rewards including step penalties
            episode_steps += 1
            global_step += 1
        
        ppo_trainer.log_episode(episode_reward, episode_steps)
        
        if len(ppo_trainer.buffer) >= config.ppo.rollout_steps:
            update_metrics = ppo_trainer.update()
            
            if update_metrics:
                for key, value in update_metrics.items():
                    writer.add_scalar(f"train/{key}", value, global_step)
        
        if episode % 10 == 0:
            stats = ppo_trainer.get_stats()
            if stats:
                for key, value in stats.items():
                    writer.add_scalar(f"episode/{key}", value, episode)
            
            # Log action distribution every 10 episodes
            total_actions = sum(action_counts.values())
            if total_actions > 0:
                for action_id, count in action_counts.items():
                    writer.add_scalar(f"train/action_{action_id}_frac", count / total_actions, episode)
                action_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # Reset counts
        
        if episode > 0 and episode % config.eval_interval == 0:
            eval_queries = test_queries[:100]
            eval_result = evaluator.evaluate_policy(eval_queries, deterministic=True)
            
            writer.add_scalar("eval/ndcg_at_10", eval_result.ndcg_at_10, episode)
            writer.add_scalar("eval/avg_steps", eval_result.avg_steps, episode)
            
            for action_id, fraction in eval_result.action_distribution.items():
                writer.add_scalar(f"eval/action_{action_id}", fraction, episode)
        
        if episode > 0 and episode % config.save_interval == 0:
            checkpoint_path = config.checkpoint_dir / f"policy_{episode}.pt"
            torch.save({
                "episode": episode,
                "policy_state_dict": policy.state_dict(),
                "optimizer_state_dict": ppo_trainer.optimizer.state_dict(),
            }, checkpoint_path)
    
    final_path = config.checkpoint_dir / "policy_final.pt"
    torch.save({
        "episode": config.total_episodes,
        "policy_state_dict": policy.state_dict(),
    }, final_path)
    
    writer.close()
    
    return policy, test_queries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="nfcorpus")
    parser.add_argument("--device", type=str, default=None, help="cuda, mps, or cpu (auto-detects if not specified)")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--entropy-coef", type=float, default=0.05, help="Entropy coefficient for exploration")
    parser.add_argument("--step-penalty", type=float, default=0.02, help="Penalty per step to encourage efficiency")
    args = parser.parse_args()
    
    config = TrainConfig(
        dataset_name=args.dataset,
        device=args.device,
        total_episodes=args.episodes,
        seed=args.seed,
        log_dir=Path(args.log_dir),
        checkpoint_dir=Path(args.checkpoint_dir)
    )
    config.ppo.lr = args.lr
    config.ppo.batch_size = args.batch_size
    config.ppo.entropy_coef = args.entropy_coef
    config.env.max_steps = args.max_steps
    config.env.step_penalty = args.step_penalty
    
    train(config)


if __name__ == "__main__":
    main()

