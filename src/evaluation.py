import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm

from .environment import SearchEnv, ACTION_SEARCH, ACTION_NARROW, ACTION_BROAD, ACTION_TERMINATE
from .policy import PolicyNetwork, PolicyState
from .query_gen import SyntheticQuery


@dataclass
class EvalResult:
    ndcg_at_10: float
    mrr_at_10: float
    avg_steps: float
    action_distribution: Dict[int, float]
    difficulty_breakdown: Dict[str, Dict[str, float]]


class HeuristicAgent:
    def __init__(self, strategy: str = "double_reform"):
        self.strategy = strategy
        self.step_count = 0
    
    def reset(self):
        self.step_count = 0
    
    def get_action(self, obs: Dict) -> int:
        self.step_count += 1
        
        if self.strategy == "single_pass":
            if self.step_count == 1:
                return ACTION_SEARCH
            return ACTION_TERMINATE
        
        elif self.strategy == "double_reform":
            if self.step_count == 1:
                return ACTION_SEARCH
            elif self.step_count == 2:
                return ACTION_NARROW
            elif self.step_count == 3:
                return ACTION_SEARCH
            elif self.step_count == 4:
                return ACTION_BROAD
            elif self.step_count == 5:
                return ACTION_SEARCH
            return ACTION_TERMINATE
        
        return ACTION_TERMINATE


class Evaluator:
    def __init__(
        self,
        env: SearchEnv,
        policy: PolicyNetwork,
        device: str = "cuda"
    ):
        self.env = env
        self.policy = policy
        self.device = device
    
    def evaluate_policy(
        self,
        test_queries: List[SyntheticQuery],
        deterministic: bool = True,
        verbose: bool = False
    ) -> EvalResult:
        self.policy.eval()
        
        rewards = []
        steps = []
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        difficulty_results = {"easy": [], "medium": [], "hard": []}
        
        with torch.no_grad():
            for query in tqdm(test_queries, disable=not verbose, desc="Evaluating"):
                obs, info = self.env.reset(options={"query": query})
                
                policy_state = PolicyState(self.policy.hidden_dim, self.device)
                
                episode_reward = 0
                episode_steps = 0
                done = False
                
                while not done:
                    state = policy_state.get_state()
                    action, _, _ = self.policy.get_action(state, deterministic=deterministic)
                    
                    action_counts[action] += 1
                    
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated
                    
                    policy_state.update(
                        self.policy.trajectory_encoder,
                        obs["query_embedding"],
                        action,
                        obs["result_embedding"]
                    )
                    
                    episode_reward = reward
                    episode_steps += 1
                
                rewards.append(episode_reward)
                steps.append(episode_steps)
                
                if episode_reward > 0.7:
                    difficulty_results["easy"].append(episode_reward)
                elif episode_reward > 0.4:
                    difficulty_results["medium"].append(episode_reward)
                else:
                    difficulty_results["hard"].append(episode_reward)
        
        total_actions = sum(action_counts.values())
        action_dist = {k: v / total_actions for k, v in action_counts.items()}
        
        difficulty_breakdown = {}
        for diff, res in difficulty_results.items():
            if res:
                difficulty_breakdown[diff] = {
                    "count": len(res),
                    "mean_ndcg": np.mean(res),
                    "std_ndcg": np.std(res)
                }
        
        return EvalResult(
            ndcg_at_10=np.mean(rewards),
            mrr_at_10=0.0,
            avg_steps=np.mean(steps),
            action_distribution=action_dist,
            difficulty_breakdown=difficulty_breakdown
        )
    
    def evaluate_baseline(
        self,
        test_queries: List[SyntheticQuery],
        strategy: str = "single_pass",
        verbose: bool = False
    ) -> EvalResult:
        agent = HeuristicAgent(strategy=strategy)
        
        rewards = []
        steps = []
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        difficulty_results = {"easy": [], "medium": [], "hard": []}
        
        for query in tqdm(test_queries, disable=not verbose, desc=f"Evaluating {strategy}"):
            obs, info = self.env.reset(options={"query": query})
            agent.reset()
            
            episode_reward = 0
            episode_steps = 0
            done = False
            
            while not done:
                action = agent.get_action(obs)
                action_counts[action] += 1
                
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                episode_reward = reward
                episode_steps += 1
            
            rewards.append(episode_reward)
            steps.append(episode_steps)
            
            if episode_reward > 0.7:
                difficulty_results["easy"].append(episode_reward)
            elif episode_reward > 0.4:
                difficulty_results["medium"].append(episode_reward)
            else:
                difficulty_results["hard"].append(episode_reward)
        
        total_actions = sum(action_counts.values())
        action_dist = {k: v / total_actions for k, v in action_counts.items()}
        
        difficulty_breakdown = {}
        for diff, res in difficulty_results.items():
            if res:
                difficulty_breakdown[diff] = {
                    "count": len(res),
                    "mean_ndcg": np.mean(res),
                    "std_ndcg": np.std(res)
                }
        
        return EvalResult(
            ndcg_at_10=np.mean(rewards),
            mrr_at_10=0.0,
            avg_steps=np.mean(steps),
            action_distribution=action_dist,
            difficulty_breakdown=difficulty_breakdown
        )
    
    def compare_all(
        self,
        test_queries: List[SyntheticQuery],
        verbose: bool = True
    ) -> Dict[str, EvalResult]:
        results = {}
        
        results["rl_agent"] = self.evaluate_policy(test_queries, verbose=verbose)
        results["single_pass"] = self.evaluate_baseline(test_queries, "single_pass", verbose=verbose)
        results["heuristic"] = self.evaluate_baseline(test_queries, "double_reform", verbose=verbose)
        
        return results
    
    def print_comparison(self, results: Dict[str, EvalResult]):
        print("\n" + "=" * 70)
        print(f"{'Method':<20} {'nDCG@10':<12} {'Avg Steps':<12} {'Actions'}")
        print("=" * 70)
        
        for name, result in results.items():
            action_str = " ".join([f"A{k}:{v:.2f}" for k, v in result.action_distribution.items()])
            print(f"{name:<20} {result.ndcg_at_10:<12.4f} {result.avg_steps:<12.2f} {action_str}")
        
        print("=" * 70)
        
        print("\nDifficulty Breakdown:")
        for name, result in results.items():
            print(f"\n{name}:")
            for diff, stats in result.difficulty_breakdown.items():
                print(f"  {diff}: count={stats['count']}, nDCG={stats['mean_ndcg']:.4f} +/- {stats['std_ndcg']:.4f}")

