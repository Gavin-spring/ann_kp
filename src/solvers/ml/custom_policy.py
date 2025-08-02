# src/solvers/ml/custom_policy.py

import gymnasium as gym
import torch
from torch import nn
from typing import Dict, List, Tuple

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import CategoricalDistribution

# --- Encoder ---
class KnapsackEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, embedding_dim: int = 128, nhead: int = 4, num_layers: int = 2):
        # 输出的特征维度是聚合后的向量维度
        super().__init__(observation_space, features_dim=embedding_dim)
        
        max_items = observation_space["items"].shape[0]
        item_feature_dim = observation_space["items"].shape[1]

        self.item_embedder = nn.Linear(item_feature_dim, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.positional_encoding = nn.Parameter(torch.randn(1, max_items, embedding_dim))
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        items_obs = observations["items"]
        item_embeddings = self.item_embedder(items_obs)
        item_embeddings += self.positional_encoding[:, :item_embeddings.size(1), :]
        
        # (batch, max_n, embed_dim)
        context = self.transformer_encoder(item_embeddings)
        
        # 聚合特征用于ValueNet
        pooled_features = torch.mean(context, dim=1)
        
        # **同时返回序列(context)和聚合特征(pooled_features)**
        return context, pooled_features

# --- Decoder ---
class PointerDecoder(nn.Module):
    def __init__(self, embedding_dim: int, n_glimpses: int = 1):
        super().__init__()
        # 这是一个简化的版本，你可以将原来复杂的Glimpse和Attention层代码搬过来
        # 关键在于，它的输入是Encoder的输出，输出是指向每个物品的logits
        self.project_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.v = nn.Parameter(torch.randn(embedding_dim), requires_grad=True)

    def forward(self, context: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        # context: (batch, max_n, embed_dim) - from Encoder
        # query: (batch, embed_dim) - 代表当前状态的查询向量 (例如context的平均池化)
        
        # (batch, max_n, embed_dim)
        projected_context = self.project_context(context)
        # (batch, 1, embed_dim)
        projected_query = self.project_query(query).unsqueeze(1)
        
        # (batch, max_n)
        scores = torch.sum(self.v * torch.tanh(projected_context + projected_query), dim=-1)
        return scores

# --- Actor-Critic Policy ---
class KnapsackActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, lr_schedule, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        self.mlp_extractor = None

    def _build(self, lr_schedule):
        self.action_net = PointerDecoder(self.features_extractor.features_dim)
        self.value_net = nn.Linear(self.features_extractor.features_dim, 1)
        self.action_dist = CategoricalDistribution(self.action_space.n)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, obs: Dict[str, torch.Tensor], deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context, pooled_features = self.extract_features(obs)
        values = self.value_net(pooled_features)
        
        action_logits = self.action_net(context, pooled_features)
        action_logits[~obs["mask"].bool()] = -torch.inf
        
        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        
        return actions, values, log_prob

    def evaluate_actions(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context, pooled_features = self.extract_features(obs)
        values = self.value_net(pooled_features)
        
        action_logits = self.action_net(context, pooled_features)
        action_logits[~obs["mask"].bool()] = -torch.inf
        
        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        
        return values, log_prob, entropy

    def predict_values(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        _context, pooled_features = self.extract_features(obs)
        return self.value_net(pooled_features)

    def _predict(self, observation: Dict[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:

        context, pooled_features = self.extract_features(observation)
        
        action_logits = self.action_net(context, pooled_features)
        action_logits[~observation["mask"].bool()] = -torch.inf
        
        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        return distribution.get_actions(deterministic=deterministic)