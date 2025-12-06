import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from .retriever import Retriever
from .reward import RewardModel
from .reformulator import Reformulator
from .query_gen import SyntheticQuery


ACTION_SEARCH = 0
ACTION_NARROW = 1
ACTION_BROAD = 2
ACTION_TERMINATE = 3


@dataclass
class StepInfo:
    query: str
    query_embedding: np.ndarray
    action: int
    result_embedding: np.ndarray
    doc_ids: List[str]
    documents: List[Dict]


class SearchEnv(gym.Env):
    def __init__(
        self,
        retriever: Retriever,
        reward_model: RewardModel,
        reformulator: Reformulator,
        query_bank: List[SyntheticQuery],
        max_steps: int = 5,
        top_k: int = 10,
        device: str = "cuda",
        step_penalty: float = 0.0
    ):
        super().__init__()
        
        self.retriever = retriever
        self.reward_model = reward_model
        self.reformulator = reformulator
        self.query_bank = query_bank
        self.max_steps = max_steps
        self.top_k = top_k
        self.device = device
        self.step_penalty = step_penalty
        
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict({
            "query_embedding": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(retriever.embedding_dim,),
                dtype=np.float32
            ),
            "result_embedding": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(retriever.embedding_dim,),
                dtype=np.float32
            ),
            "step": spaces.Discrete(max_steps + 1)
        })
        
        self.current_query: Optional[SyntheticQuery] = None
        self.current_query_str: str = ""
        self.current_step: int = 0
        self.trajectory: List[StepInfo] = []
        self.current_docs: List[Dict] = []
        self.current_doc_ids: List[str] = []
        self.current_result_embedding: np.ndarray = np.zeros(retriever.embedding_dim)
        self.terminated: bool = False
        self.has_searched: bool = False  # Track if agent has searched at least once
    
    def get_valid_actions(self) -> List[int]:
        """Return list of valid actions based on current state."""
        if not self.has_searched:
            # Must search first before reformulating or terminating
            return [ACTION_SEARCH]
        return [ACTION_SEARCH, ACTION_NARROW, ACTION_BROAD, ACTION_TERMINATE]
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        super().reset(seed=seed)
        
        if options and "query" in options:
            self.current_query = options["query"]
        else:
            idx = self.np_random.integers(0, len(self.query_bank))
            self.current_query = self.query_bank[idx]
        
        self.current_query_str = self.current_query.query
        self.current_step = 0
        self.trajectory = []
        self.current_docs = []
        self.current_doc_ids = []
        self.current_result_embedding = np.zeros(self.retriever.embedding_dim)
        self.terminated = False
        self.has_searched = False
        
        query_embedding = self.retriever.encode_query(self.current_query_str)
        
        obs = {
            "query_embedding": query_embedding.astype(np.float32),
            "result_embedding": self.current_result_embedding.astype(np.float32),
            "step": self.current_step
        }
        
        info = {
            "query": self.current_query_str,
            "relevant_doc_ids": self.current_query.relevant_doc_ids,
            "query_type": self.current_query.query_type
        }
        
        return obs, info
    
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        if self.terminated:
            raise RuntimeError("Episode already terminated")
        
        query_embedding = self.retriever.encode_query(self.current_query_str)
        
        reward = -self.step_penalty  # Small penalty per step to encourage efficiency
        terminated = False
        truncated = False
        
        if action == ACTION_SEARCH:
            result = self.retriever.search(self.current_query_str, self.top_k)
            self.current_docs = result.documents
            self.current_doc_ids = result.doc_ids
            self.current_result_embedding = self.retriever.get_aggregated_embedding(
                result.doc_ids[:3]
            )
            self.has_searched = True        
        elif action == ACTION_NARROW:
            if self.current_docs:
                new_query = self.reformulator.narrow(self.current_query_str, self.current_docs)
                self.current_query_str = new_query
                query_embedding = self.retriever.encode_query(self.current_query_str)
        
        elif action == ACTION_BROAD:
            if self.current_docs:
                new_query = self.reformulator.broaden(self.current_query_str, self.current_docs)
                self.current_query_str = new_query
                query_embedding = self.retriever.encode_query(self.current_query_str)
        
        elif action == ACTION_TERMINATE:
            terminated = True
            self.terminated = True
            
            if self.current_docs:
                ndcg, scores = self.reward_model.compute_ndcg(
                    self.current_query.query,
                    self.current_docs,
                    k=self.top_k
                )
                reward = ndcg + reward  # Add nDCG to step penalty (reward is negative)
            else:
                # Penalize terminating without any search results
                reward = -0.5  # Strong penalty for empty results
        
        step_info = StepInfo(
            query=self.current_query_str,
            query_embedding=query_embedding,
            action=action,
            result_embedding=self.current_result_embedding.copy(),
            doc_ids=self.current_doc_ids.copy(),
            documents=self.current_docs.copy()
        )
        self.trajectory.append(step_info)
        
        self.current_step += 1
        
        if self.current_step >= self.max_steps and not terminated:
            truncated = True
            if self.current_docs:
                ndcg, scores = self.reward_model.compute_ndcg(
                    self.current_query.query,
                    self.current_docs,
                    k=self.top_k
                )
                reward = ndcg
        
        obs = {
            "query_embedding": query_embedding.astype(np.float32),
            "result_embedding": self.current_result_embedding.astype(np.float32),
            "step": self.current_step
        }
        
        info = {
            "query": self.current_query_str,
            "original_query": self.current_query.query,
            "action": action,
            "num_docs": len(self.current_docs),
            "trajectory_length": len(self.trajectory)
        }
        
        return obs, reward, terminated, truncated, info
    
    def get_trajectory_tensors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.trajectory:
            return (
                np.zeros((1, self.retriever.embedding_dim)),
                np.zeros(1, dtype=np.int64),
                np.zeros((1, self.retriever.embedding_dim))
            )
        
        query_embs = np.stack([s.query_embedding for s in self.trajectory])
        actions = np.array([s.action for s in self.trajectory], dtype=np.int64)
        result_embs = np.stack([s.result_embedding for s in self.trajectory])
        
        return query_embs, actions, result_embs

