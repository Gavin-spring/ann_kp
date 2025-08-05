# src/solvers/ml/custom_policy.py

import gymnasium as gym
import torch
from torch import nn
from typing import Dict, List, Tuple

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import CategoricalDistribution
from src.utils.config_loader import cfg

# --- Encoder ---
class KnapsackEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, embedding_dim: int = 128, nhead: int = 4, num_layers: int = 2):
        super().__init__(observation_space, features_dim=embedding_dim)

        item_feature_dim = cfg.ml.rl.ppo.hyperparams.item_feature_dim  # weight, value
        self.max_possible_n = cfg.ml.rl.ppo.hyperparams.eval_max_n

        self.item_embedder = nn.Linear(item_feature_dim, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.positional_encoding = nn.Parameter(torch.randn(1, self.max_possible_n, embedding_dim))
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        items_obs = observations["items"]
        batch_size, seq_len, _ = items_obs.shape
        item_embeddings = self.item_embedder(items_obs)
        item_embeddings += self.positional_encoding[:, :seq_len, :]
        
        # (batch, max_n, embed_dim)
        context = self.transformer_encoder(item_embeddings)
        pooled_features = torch.mean(context, dim=1)

        return context, pooled_features

# --- Decoder ---
class PointerDecoder(nn.Module):
    def __init__(self, embedding_dim: int, n_glimpses: int = 1):
        super().__init__()
        self.n_glimpses = n_glimpses
        self.glimpse_attention = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.pointer_attention = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.project_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        # self.project_query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.v = nn.Parameter(torch.randn(embedding_dim), requires_grad=True)

    def forward(self, context: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        # context: (batch, max_n, embed_dim) - from Encoder
        # query: (batch, embed_dim)

        projected_context = self.project_context(context)
        for _ in range(self.n_glimpses):
            # (batch, 1, embed_dim)
            projected_glimpse_query = self.glimpse_attention(query).unsqueeze(1)
            
            # (batch, max_n)
            glimpse_scores = torch.sum(self.v * torch.tanh(projected_context + projected_glimpse_query), dim=-1)
            glimpse_weights = torch.softmax(glimpse_scores, dim=1)
            
            # 用注意力权重重新聚合context，得到新的query
            # bmm: (batch, 1, max_n) @ (batch, max_n, embed_dim) -> (batch, 1, embed_dim)
            query = torch.bmm(glimpse_weights.unsqueeze(1), context).squeeze(1)

        # 2. Pointer Phase: 使用最终精炼过的query进行决策
        projected_pointer_query = self.pointer_attention(query).unsqueeze(1)
        final_scores = torch.sum(self.v * torch.tanh(projected_context + projected_pointer_query), dim=-1)
        
        return final_scores
        # # (batch, max_n, embed_dim)
        # projected_context = self.project_context(context)
        # # (batch, 1, embed_dim)
        # projected_query = self.project_query(query).unsqueeze(1)
        
        # # (batch, max_n)
        # scores = torch.sum(self.v * torch.tanh(projected_context + projected_query), dim=-1)
        # return scores

# --- Critic ---
# Critic should share the same Encoder as Actor
class CriticNetwork(nn.Module):
    def __init__(self, embedding_dim: int, n_process_block_iters: int = 3):
        super().__init__()
        
        self.n_process_block_iters = n_process_block_iters
        
        # 用于状态精炼的注意力模块 (Process Block)
        self.process_block = PointerDecoder(embedding_dim) # 我们可以复用PointerDecoder作为注意力层
        
        # 最终输出价值的MLP
        self.value_decoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )

    def forward(self, context: torch.Tensor, pooled_features: torch.Tensor) -> torch.Tensor:
        # pooled_features作为初始的"状态"，进行迭代精炼
        process_block_state = pooled_features
        for _ in range(self.n_process_block_iters):
            # 用当前状态作为query，在所有物品的context上做注意力
            attention_logits = self.process_block(context, process_block_state)
            attention_weights = torch.softmax(attention_logits, dim=1)
            
            # 根据注意力权重，重新聚合context，得到新的、更精炼的状态
            # bmm: (batch, 1, max_n) @ (batch, max_n, embed_dim) -> (batch, 1, embed_dim)
            process_block_state = torch.bmm(attention_weights.unsqueeze(1), context).squeeze(1)
            
        # 3. 使用最终精炼过的状态来预测价值
        return self.value_decoder(process_block_state)

# --- Actor-Critic Policy ---
class KnapsackActorCriticPolicy(ActorCriticPolicy):
    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 lr_schedule,
                 n_process_block_iters: int = 3,
                 **kwargs):
        self.n_process_block_iters = n_process_block_iters
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule):
        # Actor
        self.action_net = PointerDecoder(self.features_extractor.features_dim)

        # Critic
        features_dim = self.features_extractor.features_dim
        
        self.value_net = CriticNetwork(embedding_dim=features_dim, n_process_block_iters=self.n_process_block_iters)
        self.action_dist = CategoricalDistribution(self.action_space.n)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_action_logits_from_obs(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """从观测计算出最终的、安全的logits"""
        context, pooled_features = self.extract_features(obs)
        action_logits = self.action_net(context, pooled_features)
        
        mask = obs["mask"].bool()
        action_logits[~mask] = -torch.inf

        # 检查是否存在所有动作都被屏蔽的行
        all_masked_rows = torch.all(~mask, dim=1)
        if all_masked_rows.any():
            # 对于这些完全被屏蔽的行，我们将第一个动作的logit设为0
            # 这可以确保softmax的输出是 [1, 0, 0, ...] 而不是 [nan, nan, nan, ...]
            # 从而避免CUDA崩溃。这个动作是无效的，但可以安全地被采样。
            action_logits[all_masked_rows, 0] = 0

        return action_logits

    def forward(self, obs: Dict[str, torch.Tensor], deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # shared features encoder
        context, pooled_features = self.extract_features(obs)

        # critic
        values = self.value_net(context, pooled_features)

        # actor
        action_logits = self.action_net(context, pooled_features)

        # mask
        mask = obs["mask"].bool()
        action_logits[~mask] = -torch.inf

        all_masked_rows = torch.all(~mask, dim=1)
        if all_masked_rows.any():
            action_logits[all_masked_rows, 0] = 0

        # get results
        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        
        return actions, values, log_prob

    def evaluate_actions(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # --- 步骤 1: 只调用一次特征提取器，获取共享的特征 ---
        # context 用于 Actor (PointerDecoder)，pooled_features 用于 Critic (MLP)
        context, pooled_features = self.extract_features(obs)
        
        # --- 步骤 2: Critic 使用详细的 context 和池化后的特征计算价值 ---
        values = self.value_net(context, pooled_features)

        # --- 步骤 3: Actor 使用详细的 context 和池化的 query 计算 logits ---
        action_logits = self.action_net(context, pooled_features)
        
        # --- 步骤 4: 应用 mask 来屏蔽无效动作 ---
        mask = obs["mask"].bool()
        action_logits[~mask] = -torch.inf

        # 检查并处理所有动作都被屏蔽的特殊情况
        all_masked_rows = torch.all(~mask, dim=1)
        if all_masked_rows.any():
            action_logits[all_masked_rows, 0] = 0
            
        # --- 步骤 5: 从安全的 logits 计算 log_prob 和 entropy ---
        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        
        return values, log_prob, entropy

    def predict_values(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        context, pooled_features = self.extract_features(obs)
        return self.value_net(context, pooled_features)

    def _predict(self, observation: Dict[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:
        action_logits = self._get_action_logits_from_obs(observation)
        
        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        return distribution.get_actions(deterministic=deterministic)
