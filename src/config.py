from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class RetrieverConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    index_path: Optional[Path] = None
    top_k: int = 10
    embedding_dim: int = 384


@dataclass
class PolicyConfig:
    hidden_dim: int = 256
    gru_layers: int = 1
    action_dim: int = 4
    embedding_dim: int = 384
    dropout: float = 0.1


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.05  # Increased from 0.01 to encourage exploration
    max_grad_norm: float = 0.5
    batch_size: int = 64
    epochs_per_update: int = 4
    rollout_steps: int = 2048


@dataclass
class EnvConfig:
    max_steps: int = 5
    reward_scale: float = 1.0
    step_penalty: float = 0.02  # Small penalty per step to encourage efficiency


@dataclass
class CurriculumConfig:
    """Curriculum learning settings to focus on hard queries."""
    enabled: bool = False
    hard_query_threshold: float = 0.6  # nDCG below this = hard query
    hard_query_ratio: float = 0.7  # Fraction of hard queries in training batch
    warmup_episodes: int = 1000  # Start curriculum after this many episodes


@dataclass
class RewardConfig:
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    model_name_mps: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_length: int = 512
    batch_size: int = 8
    batch_size_mps: int = 4
    temperature: float = 0.1
    use_quantization: bool = False


@dataclass
class ReformulatorConfig:
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    model_name_mps: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_new_tokens: int = 128
    temperature: float = 0.7
    use_quantization: bool = False


@dataclass
class QueryGenConfig:
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    model_name_mps: str = "Qwen/Qwen2.5-1.5B-Instruct"
    queries_per_doc: int = 3
    batch_size: int = 4
    batch_size_mps: int = 2
    use_quantization: bool = False


@dataclass
class TrainConfig:
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    reformulator: ReformulatorConfig = field(default_factory=ReformulatorConfig)
    query_gen: QueryGenConfig = field(default_factory=QueryGenConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    
    total_episodes: int = 10000
    eval_interval: int = 500
    save_interval: int = 1000
    log_dir: Path = field(default_factory=lambda: Path("runs"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    dataset_name: str = "nfcorpus"
    device: str = "cuda"
    seed: int = 42
    
    def optimize_for_device(self):
        from .utils import get_device, get_optimal_batch_size, is_apple_silicon
        
        self.device = get_device(self.device)
        
        if self.device == "mps" or (self.device == "cpu" and is_apple_silicon()):
            self.reward.model_name = self.reward.model_name_mps
            self.reward.batch_size = self.reward.batch_size_mps
            self.reformulator.model_name = self.reformulator.model_name_mps
            self.query_gen.model_name = self.query_gen.model_name_mps
            self.query_gen.batch_size = self.query_gen.batch_size_mps
            self.ppo.batch_size = get_optimal_batch_size(self.device, self.ppo.batch_size)

