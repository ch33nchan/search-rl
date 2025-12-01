import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple, List, Optional
import numpy as np


class TrajectoryEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 384,
        hidden_dim: int = 256,
        action_dim: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        
        self.query_proj = nn.Linear(embedding_dim, hidden_dim)
        self.result_proj = nn.Linear(embedding_dim, hidden_dim)
        self.action_embed = nn.Embedding(action_dim, hidden_dim)
        
        self.step_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
    
    def forward(
        self,
        query_embeddings: torch.Tensor,
        action_ids: torch.Tensor,
        result_embeddings: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = query_embeddings.shape
        
        query_feat = self.query_proj(query_embeddings)
        result_feat = self.result_proj(result_embeddings)
        action_feat = self.action_embed(action_ids)
        
        step_input = torch.cat([query_feat, action_feat, result_feat], dim=-1)
        step_feat = self.step_encoder(step_input)
        
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                step_feat, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, hidden = self.gru(packed)
        else:
            _, hidden = self.gru(step_feat)
        
        state = self.output_proj(hidden[-1])
        return state
    
    def encode_single_step(
        self,
        query_embedding: torch.Tensor,
        action_id: torch.Tensor,
        result_embedding: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query_feat = self.query_proj(query_embedding.unsqueeze(0).unsqueeze(0))
        result_feat = self.result_proj(result_embedding.unsqueeze(0).unsqueeze(0))
        action_feat = self.action_embed(action_id.unsqueeze(0).unsqueeze(0))
        
        step_input = torch.cat([query_feat, action_feat, result_feat], dim=-1)
        step_feat = self.step_encoder(step_input)
        
        output, hidden = self.gru(step_feat, hidden)
        state = self.output_proj(output.squeeze(1))
        
        return state, hidden


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 384,
        hidden_dim: int = 256,
        action_dim: int = 4,
        gru_layers: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        self.trajectory_encoder = TrajectoryEncoder(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            num_layers=gru_layers,
            dropout=dropout
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        
        nn.init.orthogonal_(self.policy_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)
    
    def forward(
        self,
        query_embeddings: torch.Tensor,
        action_ids: torch.Tensor,
        result_embeddings: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        state = self.trajectory_encoder(
            query_embeddings, action_ids, result_embeddings, lengths
        )
        
        action_logits = self.policy_head(state)
        value = self.value_head(state).squeeze(-1)
        
        return action_logits, value
    
    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        action_logits = self.policy_head(state)
        value = self.value_head(state).squeeze(-1)
        
        dist = Categorical(logits=action_logits)
        
        if deterministic:
            action = action_logits.argmax(dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob, value
    
    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_logits = self.policy_head(states)
        values = self.value_head(states).squeeze(-1)
        
        dist = Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values, entropy


class PolicyState:
    def __init__(self, hidden_dim: int, device: str = "cuda"):
        self.hidden_dim = hidden_dim
        self.device = device
        self.reset()
    
    def reset(self):
        self.query_embeddings: List[np.ndarray] = []
        self.action_ids: List[int] = []
        self.result_embeddings: List[np.ndarray] = []
        self.gru_hidden: Optional[torch.Tensor] = None
        self.current_state: Optional[torch.Tensor] = None
    
    def update(
        self,
        encoder: TrajectoryEncoder,
        query_embedding: np.ndarray,
        action_id: int,
        result_embedding: np.ndarray
    ):
        self.query_embeddings.append(query_embedding)
        self.action_ids.append(action_id)
        self.result_embeddings.append(result_embedding)
        
        q_tensor = torch.tensor(query_embedding, dtype=torch.float32, device=self.device)
        a_tensor = torch.tensor(action_id, dtype=torch.long, device=self.device)
        r_tensor = torch.tensor(result_embedding, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            self.current_state, self.gru_hidden = encoder.encode_single_step(
                q_tensor, a_tensor, r_tensor, self.gru_hidden
            )
    
    def get_state(self) -> torch.Tensor:
        if self.current_state is None:
            return torch.zeros(1, self.hidden_dim, device=self.device)
        return self.current_state

