"""PressLight adapter inherited from LLMTSCS-main.

Uses shared DQN parameters and incoming/outgoing lane-pressure observations."""

from agent.drl_based.base_drl import BaseDRL
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


class PressLightNetwork(nn.Module):
    """Shared fully connected network with flattened features and one Q-value per phase."""

    def __init__(self, input_dims_dict, num_phase, d_dense=20, n_layer=2):
        """Initialize feature dimensions, phase count, hidden width and hidden-layer count."""
        super(PressLightNetwork, self).__init__()
        self.num_phase = num_phase
        self.d_dense = d_dense
        self.n_layer = n_layer
        self.input_dims_dict = input_dims_dict


        total_input_dim = sum(input_dims_dict.values())


        layers = []
        in_dim = total_input_dim
        for _ in range(n_layer):
            layers.append(nn.Linear(in_dim, d_dense))
            layers.append(nn.ReLU())
            in_dim = d_dense
        self.shared_mlp = nn.Sequential(*layers)


        self.output_layer = nn.Linear(d_dense, num_phase)

        print(f"[PressLightNetwork] total_input_dim={total_input_dim}, num_phase={num_phase}, "
              f"d_dense={d_dense}, n_layer={n_layer}")

    def forward(self, x_dict):
        """Evaluate a feature-name to (batch, feature_dim) tensor dictionary."""

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


        h = self.shared_mlp(x)


        q_values = self.output_layer(h)
        return q_values


class PressLight(BaseDRL):
    """Parameter-sharing DQN agent with incoming and outgoing lane-pressure observations."""

    def __init__(self, config, env, idx):
        self.pre_init_config = config
        self.pre_init_env = env
        self.pre_init_idx = idx
        super().__init__(config, env, idx)

        agent_config = config.get("PressLight", {})


        obs_features = agent_config.get("observation_feature_list", ["inter_2_pressure"])



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

        # Use the inherited RMSprop optimizer.
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

            x_dict = {}
            for feat_name in self.obs_features:
                if isinstance(obs, list):

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


            for key in x_dict:
                if x_dict[key].dim() == 1:
                    x_dict[key] = x_dict[key].unsqueeze(0)
                elif x_dict[key].dim() > 2:

                    x_dict[key] = x_dict[key].view(1, -1)

            q_values = self.network_local(x_dict)
            # Read epsilon at each decision so evaluation can disable exploration.
            current_epsilon = self.cur_agent.get("epsilon", 0.1)
            if random.random() < current_epsilon:

                action = random.randint(0, self.num_phase - 1)
            else:

                action = torch.argmax(q_values, dim=1).item()

        # Restore training mode only during training.
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
            # Scale rewards by the inherited NORMAL_FACTOR (default 20).
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
