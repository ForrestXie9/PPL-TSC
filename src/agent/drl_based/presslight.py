"""
PressLight Agent for Traffic Signal Control
迁移自 LLMTSCS-main：基于 pressure 信号的 DQN，参数共享，与论文实现一致。
使用 inlane + outlane pressure 作为观测特征。
"""

from agent.drl_based.base_drl import BaseDRL
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


class PressLightNetwork(nn.Module):
    """
    PressLight Network（对齐原仓库 SimpleDQN 的多层 MLP 结构）：
    - 输入：多个特征（phase、pressure 等），先分别 flatten 再拼接
    - 结构：若干层共享全连接（ReLU），最后一层线性输出 num_phase 个 Q 值
    """

    def __init__(self, input_dims_dict, num_phase, d_dense=20, n_layer=2):
        """
        Args:
            input_dims_dict: dict，每个特征的维度，如 {"cur_phase": 8, "pressure": 12}
            num_phase: 相位数量（动作维度）
            d_dense: 隐层维度（对应 D_DENSE）
            n_layer: 隐层层数（对应 N_LAYER）
        """
        super(PressLightNetwork, self).__init__()
        self.num_phase = num_phase
        self.d_dense = d_dense
        self.n_layer = n_layer
        self.input_dims_dict = input_dims_dict

        # 计算总输入维度
        total_input_dim = sum(input_dims_dict.values())

        # 多层共享全连接（SimpleDQN 风格）
        layers = []
        in_dim = total_input_dim
        for _ in range(n_layer):
            layers.append(nn.Linear(in_dim, d_dense))
            layers.append(nn.ReLU())
            in_dim = d_dense
        self.shared_mlp = nn.Sequential(*layers)

        # 最后一层线性输出：每个相位一个 Q 值
        self.output_layer = nn.Linear(d_dense, num_phase)

        print(f"[PressLightNetwork] total_input_dim={total_input_dim}, num_phase={num_phase}, "
              f"d_dense={d_dense}, n_layer={n_layer}")

    def forward(self, x_dict):
        """
        Args:
            x_dict: dict，key 为特征名，value 为 tensor (batch, feature_dim)
        
        Returns:
            q_values: (batch, num_phase)
        """
        # 拼接所有特征
        features = []
        device = next(self.parameters()).device
        
        for key in sorted(self.input_dims_dict.keys()):
            if key in x_dict:
                feat = x_dict[key]
                if isinstance(feat, np.ndarray):
                    feat = torch.from_numpy(feat).float().to(device)
                elif not isinstance(feat, torch.Tensor):
                    feat = torch.tensor(feat, dtype=torch.float32, device=device)
                else:
                    feat = feat.to(device)
                
                # 确保特征是 2D: (batch, feature_dim)
                if feat.dim() == 0:
                    feat = feat.unsqueeze(0).unsqueeze(0)
                elif feat.dim() == 1:
                    feat = feat.unsqueeze(0)
                
                # Flatten to (batch, -1)
                feat = feat.view(feat.size(0), -1)
                features.append(feat)
        
        if not features:
            raise ValueError("No features provided")
        
        x = torch.cat(features, dim=1)

        # 共享 MLP
        h = self.shared_mlp(x)

        # 线性输出 (batch, num_phase)
        q_values = self.output_layer(h)
        return q_values


class PressLight(BaseDRL):
    """
    PressLight Agent（迁移自 LLMTSCS）：
    使用 pressure-based 观测（inlane pressure + outlane pressure），基于参数共享的 DQN。
    """

    def __init__(self, config, env, idx):
        self.pre_init_config = config
        self.pre_init_env = env
        self.pre_init_idx = idx
        super().__init__(config, env, idx)

        agent_config = config.get("PressLight", {})

        # 根据 observation_feature_list 构建输入维度字典
        obs_features = agent_config.get("observation_feature_list", ["inter_2_pressure"])
        
        # 获取实际的观测维度（通过获取一次观测来确定）
        # 从环境获取初始观测来推断特征维度
        dummy_obs = env.reset()
        actual_obs = dummy_obs[idx]
        
        input_dims_dict = {}
        for i, feat_name in enumerate(obs_features):
            if i < len(actual_obs):
                feat_data = actual_obs[i]
                if isinstance(feat_data, torch.Tensor):
                    feat_dim = feat_data.numel()
                elif isinstance(feat_data, np.ndarray):
                    feat_dim = feat_data.size
                elif isinstance(feat_data, (list, tuple)):
                    feat_dim = len(feat_data)
                else:
                    feat_dim = 1
                input_dims_dict[feat_name] = feat_dim
            else:
                # 如果观测数量少于特征列表，使用默认值
                # 如果观测数量少于特征列表，使用默认值
                input_dims_dict[feat_name] = 12

        print(f"[PressLight Agent {idx}] Detected input dimensions: {input_dims_dict}")

        self.network_local = PressLightNetwork(
            input_dims_dict=input_dims_dict,
            num_phase=self.num_phase,
            d_dense=agent_config.get("d_dense", 20),
            n_layer=agent_config.get("n_layer", 2),
        )
        self.network_target = PressLightNetwork(
            input_dims_dict=input_dims_dict,
            num_phase=self.num_phase,
            d_dense=agent_config.get("d_dense", 20),
            n_layer=agent_config.get("n_layer", 2),
        )
        
        self.network_local = self.network_local.to(self.config["device"])
        self.network_target = self.network_target.to(self.config["device"])

        from utilities.utils import copy_model_params
        copy_model_params(source_model=self.network_local, target_model=self.network_target)

        # 对齐原仓库：RMSprop 优化器
        self.network_optim = torch.optim.RMSprop(
            self.network_local.parameters(),
            lr=self.cur_agent.get("learning_rate", 0.001),
        )
        
        self.obs_features = obs_features
        print(f"[PressLight Agent {idx}] Initialized (pressure-based DQN, num_phase={self.num_phase}, device={self.config['device']})")

    def reset(self):
        self.current_phase = 0
        self.replay_buffer.reset()

    def pick_action(self, n_obs, on_training):
        obs = n_obs[self.idx]
        self.network_local.eval()
        with torch.no_grad():
            # 构建特征字典
            x_dict = {}
            for feat_name in self.obs_features:
                if isinstance(obs, list):
                    # obs 是 list 格式
                    feat_idx = self.obs_features.index(feat_name)
                    if feat_idx < len(obs):
                        feat = obs[feat_idx]
                        if isinstance(feat, np.ndarray):
                            x_dict[feat_name] = torch.from_numpy(feat).float().to(self.config["device"])
                        elif isinstance(feat, torch.Tensor):
                            x_dict[feat_name] = feat.to(self.config["device"])
                        else:
                            x_dict[feat_name] = torch.tensor(feat, dtype=torch.float32, device=self.config["device"])
                elif isinstance(obs, dict) and feat_name in obs:
                    feat = obs[feat_name]
                    if isinstance(feat, np.ndarray):
                        x_dict[feat_name] = torch.from_numpy(feat).float().to(self.config["device"])
                    elif isinstance(feat, torch.Tensor):
                        x_dict[feat_name] = feat.to(self.config["device"])
                    else:
                        x_dict[feat_name] = torch.tensor(feat, dtype=torch.float32, device=self.config["device"])

            # 确保特征是 (batch=1, feature_dim) 格式
            for key in x_dict:
                if x_dict[key].dim() == 1:
                    x_dict[key] = x_dict[key].unsqueeze(0)
                elif x_dict[key].dim() > 2:
                    # 如果是多维的，flatten 成 (1, -1)
                    x_dict[key] = x_dict[key].view(1, -1)

            q_values = self.network_local(x_dict)
            # 动态读取epsilon（支持评估脚本动态更新epsilon为0）
            current_epsilon = self.cur_agent.get("epsilon", 0.1)
            if random.random() < current_epsilon:
                # 探索：随机选择
                action = random.randint(0, self.num_phase - 1)
            else:
                # 利用：选择最优动作
                action = torch.argmax(q_values, dim=1).item()
        
        # 只在训练时设置为train mode
        if on_training:
            self.network_local.train()
        
        self.current_phase = action
        return action

    def _compute_critic_loss(self, obs, act, rew, next_obs, done):
        # Convert batched tensor observations to dict format
        # obs and next_obs are concatenated tensors from replay buffer
        # Need to split them back into dict format
        obs_dict = self._tensor_to_dict(obs)
        next_obs_dict = self._tensor_to_dict(next_obs)
        
        q_current = self.network_local(obs_dict)
        act_indices = act.long()
        if act_indices.dim() == 1:
            act_indices = act_indices.unsqueeze(1)
        q_current = q_current.gather(1, act_indices).squeeze(1)
        
        with torch.no_grad():
            q_next_target = self.network_target(next_obs_dict)
            q_next_max = q_next_target.max(1)[0]
            # 对齐原仓库：reward 做 NORMAL_FACTOR 缩放（默认 20）
            normal_factor = float(self.cur_agent.get("normal_factor", 20.0))
            scaled_rew = rew / normal_factor
            q_target = scaled_rew + (1 - done.float()) * self.gamma * q_next_max
        
        return F.mse_loss(q_current, q_target)
    
    def _tensor_to_dict(self, obs_tensor):
        """
        Convert concatenated observation tensor or list back to dict format.
        
        Args:
            obs_tensor: (batch, total_dim) tensor OR list of tensors
        
        Returns:
            dict {feature_name: tensor (batch, feature_dim)}
        """
        if isinstance(obs_tensor, dict):
            return obs_tensor
        
        # If it's a list, assume it's already in the correct order matching obs_features
        if isinstance(obs_tensor, list):
            obs_dict = {}
            for i, feat_name in enumerate(sorted(self.network_local.input_dims_dict.keys())):
                if i < len(obs_tensor):
                    feat = obs_tensor[i]
                    if not isinstance(feat, torch.Tensor):
                        feat = torch.tensor(feat, dtype=torch.float32, device=self.config["device"])
                    obs_dict[feat_name] = feat
            return obs_dict
        
        # If it's a tensor, split it by feature dimensions
        obs_dict = {}
        start_idx = 0
        
        for feat_name in sorted(self.network_local.input_dims_dict.keys()):
            feat_dim = self.network_local.input_dims_dict[feat_name]
            end_idx = start_idx + feat_dim
            obs_dict[feat_name] = obs_tensor[:, start_idx:end_idx]
            start_idx = end_idx
        
        return obs_dict
