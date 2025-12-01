import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque


@dataclass
class Transition:
    state: torch.Tensor
    action: int
    log_prob: float
    value: float
    reward: float
    done: bool


@dataclass
class RolloutBuffer:
    states: List[torch.Tensor] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    
    def add(self, transition: Transition):
        self.states.append(transition.state)
        self.actions.append(transition.action)
        self.log_probs.append(transition.log_prob)
        self.values.append(transition.value)
        self.rewards.append(transition.reward)
        self.dones.append(transition.done)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
    
    def __len__(self):
        return len(self.states)


class PPOTrainer:
    def __init__(
        self,
        policy: nn.Module,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        epochs_per_update: int = 4,
        batch_size: int = 64,
        device: str = "cuda"
    ):
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.epochs_per_update = epochs_per_update
        self.batch_size = batch_size
        self.device = device
        
        self.buffer = RolloutBuffer()
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
    
    def store_transition(self, transition: Transition):
        self.buffer.add(transition)
    
    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_value: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        T = len(rewards)
        advantages = np.zeros(T)
        returns = np.zeros(T)
        
        gae = 0
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = last_value
                next_done = 1.0
            else:
                next_value = values[t + 1]
                next_done = dones[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - next_done) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - next_done) * gae
            advantages[t] = gae
        
        returns = advantages + values
        return advantages, returns
    
    def update(self) -> Dict[str, float]:
        if len(self.buffer) < self.batch_size:
            return {}
        
        states = torch.stack(self.buffer.states).to(self.device)
        actions = torch.tensor(self.buffer.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(self.buffer.log_probs, dtype=torch.float32, device=self.device)
        values = np.array(self.buffer.values)
        rewards = np.array(self.buffer.rewards)
        dones = np.array(self.buffer.dones, dtype=np.float32)
        
        advantages, returns = self.compute_gae(rewards, values, dones)
        
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        metrics = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": []
        }
        
        indices = np.arange(len(self.buffer))
        
        for _ in range(self.epochs_per_update):
            np.random.shuffle(indices)
            
            for start in range(0, len(indices), self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]
                
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                
                new_log_probs, new_values, entropy = self.policy.evaluate_actions(
                    batch_states, batch_actions
                )
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = nn.functional.mse_loss(new_values, batch_returns)
                
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    clip_fraction = ((ratio - 1).abs() > self.clip_epsilon).float().mean().item()
                
                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy"].append(entropy.mean().item())
                metrics["approx_kl"].append(approx_kl)
                metrics["clip_fraction"].append(clip_fraction)
        
        self.buffer.clear()
        
        return {k: np.mean(v) for k, v in metrics.items()}
    
    def log_episode(self, reward: float, length: int):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
    
    def get_stats(self) -> Dict[str, float]:
        if not self.episode_rewards:
            return {}
        return {
            "mean_reward": np.mean(self.episode_rewards),
            "std_reward": np.std(self.episode_rewards),
            "mean_length": np.mean(self.episode_lengths),
            "max_reward": np.max(self.episode_rewards),
            "min_reward": np.min(self.episode_rewards)
        }

