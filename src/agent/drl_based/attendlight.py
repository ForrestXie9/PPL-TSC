"""
AttendLight Agent for Traffic Signal Control
迁移自 LLMTSCS-main：相位分组 + 双层 Multi-Head Attention，与论文/原实现一致。
使用 inlane 观测 + phase_2_passable_lane_idx 做 per-phase 注意力再 phase-level 注意力。
"""

from agent.drl_based.base_drl import BaseDRL
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


def _build_phase_map_from_intersection(inter):
    """
    从路口的 phase_2_passable_lane_idx 得到 phase_map：
    phase_map[phase_id] = 该相位下可通行的车道索引列表（与 inlane 顺序一致）。
    """
    # phase_2_passable_lane_idx: list of length num_phase, 每个是 length=num_in_lane 的 0/1 列表
    phase_map = []
    for phase_idx in range(len(inter.phase_2_passable_lane_idx)):
        lane_indices = [
            j for j in range(len(inter.phase_2_passable_lane_idx[phase_idx]))
            if inter.phase_2_passable_lane_idx[phase_idx][j] == 1
        ]
        phase_map.append(lane_indices)
    return phase_map


class AttendLightNetwork(nn.Module):
    """
    相位分组 + 双层 MHA（对齐 LLMTSCS attendlight_agent.py）：
    - 输入: (batch, num_lane)，每条 inlane 一个特征（如车辆数）
    - 每条 lane embed 成 embed_dim
    - 每个 phase：用该 phase 下的 lane 特征做 MHA，得到该 phase 的表示
    - 对所有 phase 表示再做 MHA，然后 MLP 出 Q
    """

    def __init__(self, num_lane, num_phase, phase_map, embed_dim=32, num_heads=4, hidden_dim=20):
        super(AttendLightNetwork, self).__init__()
        self.num_lane = num_lane
        self.num_phase = num_phase
        self.phase_map = phase_map  # list of list, 不参与梯度，仅用于 forward 里索引
        self.embed_dim = embed_dim
        if embed_dim % num_heads != 0:
            embed_dim = ((embed_dim + num_heads - 1) // num_heads) * num_heads
        self.embed_dim = embed_dim

        # 每条 lane 一个标量 -> embed_dim
        self.embed = nn.Linear(1, embed_dim)

        # Per-phase: 用该 phase 内 lane 做 MHA（query=mean of lanes）
        self.phase_lane_mha = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=0.0
        )

        # Phase-level MHA
        self.phase_mha = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=0.0
        )

        self.dense1 = nn.Linear(embed_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.dense_q = nn.Linear(hidden_dim, num_phase)
        self.relu = nn.ReLU()

        # 注册 phase_map 为 buffer 以便在 forward 里用 tensor 索引（可选）
        # 这里用 list 在 CPU 上索引即可，不注册
        print(f"[AttendLightNetwork] num_lane={num_lane}, num_phase={num_phase}, embed_dim={embed_dim}, num_heads={num_heads}")

    def forward(self, x):
        """
        x: (batch, num_lane) 或 list of tensors（会 concat 成 (batch, num_lane)）
        """
        if isinstance(x, list):
            flattened = []
            for xi in x:
                if isinstance(xi, torch.Tensor):
                    if xi.dim() == 0:
                        xi = xi.unsqueeze(0).unsqueeze(0)
                    elif xi.dim() == 1:
                        xi = xi.unsqueeze(0)
                    else:
                        xi = xi.view(xi.size(0), -1)
                else:
                    xi = torch.as_tensor(xi, dtype=torch.float32, device=x[0].device if isinstance(x[0], torch.Tensor) else "cpu")
                    if xi.dim() == 1:
                        xi = xi.unsqueeze(0)
                flattened.append(xi)
            x = torch.cat(flattened, dim=1)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch = x.size(0)
        if x.size(1) != self.num_lane:
            if x.size(1) < self.num_lane:
                x = F.pad(x, (0, self.num_lane - x.size(1)))
            else:
                x = x[:, : self.num_lane]
        # (B, num_lane) -> (B, num_lane, 1) -> (B, num_lane, embed_dim)
        x = self.embed(x.unsqueeze(-1))
        device = x.device

        # 每个 phase：取对应 lane 特征，mean 为 query，做 MHA，得到 (B, 1, embed_dim)
        phase_feats = []
        for p in range(self.num_phase):
            idx = self.phase_map[p]
            if len(idx) == 0:
                phase_feats.append(torch.zeros(batch, 1, self.embed_dim, device=device, dtype=x.dtype))
                continue
            lane_p = x[:, idx, :]
            query = lane_p.mean(dim=1, keepdim=True)
            attn_out, _ = self.phase_lane_mha(query, lane_p, lane_p)
            phase_feats.append(attn_out)
        # (B, num_phase, embed_dim)
        phase_all = torch.cat(phase_feats, dim=1)

        # Phase-level MHA
        phase_attn, _ = self.phase_mha(phase_all, phase_all, phase_all)
        h = phase_attn.mean(dim=1)
        h = self.relu(self.dense1(h))
        h = self.relu(self.dense2(h))
        q_values = self.dense_q(h)
        return q_values


class AttendLight(BaseDRL):
    """
    AttendLight Agent（迁移自 LLMTSCS）：
    使用 inlane 观测 + 相位分组双层注意力，与 baseline 框架兼容。
    """

    def __init__(self, config, env, idx):
        self.pre_init_config = config
        self.pre_init_env = env
        self.pre_init_idx = idx
        super().__init__(config, env, idx)

        inter = self.inter
        num_lane = len(inter.n_in_lane_id)
        num_phase = self.num_phase
        phase_map = _build_phase_map_from_intersection(inter)

        agent_config = config.get("AttendLight", {})
        self.t_min = agent_config.get("min_phase_time", 10)
        self.heuristic_method = agent_config.get("heuristic_method", "pressure")

        self.network_local = AttendLightNetwork(
            num_lane=num_lane,
            num_phase=num_phase,
            phase_map=phase_map,
            embed_dim=agent_config.get("embed_dim", 32),
            num_heads=agent_config.get("num_heads", 4),
            hidden_dim=agent_config.get("hidden_dim", 20),
        )
        self.network_target = AttendLightNetwork(
            num_lane=num_lane,
            num_phase=num_phase,
            phase_map=phase_map,
            embed_dim=agent_config.get("embed_dim", 32),
            num_heads=agent_config.get("num_heads", 4),
            hidden_dim=agent_config.get("hidden_dim", 20),
        )
        self.network_local = self.network_local.to(self.config["device"])
        self.network_target = self.network_target.to(self.config["device"])

        from utilities.utils import copy_model_params
        copy_model_params(source_model=self.network_local, target_model=self.network_target)

        self.network_optim = torch.optim.Adam(
            self.network_local.parameters(),
            lr=self.cur_agent.get("learning_rate", 0.001),
        )
        print(f"[AttendLight Agent {idx}] Initialized (phase-based MHA, num_lane={num_lane}, phases={num_phase}, device={self.config['device']})")

    def reset(self):
        self.current_phase = 0
        self.replay_buffer.reset()

    def pick_action(self, n_obs, on_training):
        # 修复：只在相位时间>0且<t_min时保持当前相位，避免初始状态被锁定在phase 0
        # 如果current_phase_time=0（刚开始），允许根据Q值选择；只有在已经有相位时间积累但还不够t_min时才保持
        if self.inter.current_phase_time > 0 and self.inter.current_phase_time < self.t_min:
            return self.current_phase
        obs = n_obs[self.idx]
        self.network_local.eval()
        with torch.no_grad():
            if isinstance(obs, list):
                obs_tensors = [
                    o.to(self.config["device"]) if isinstance(o, torch.Tensor) else torch.tensor(o, dtype=torch.float32, device=self.config["device"])
                    for o in obs
                ]
                q_values = self.network_local(obs_tensors)
            elif isinstance(obs, torch.Tensor):
                obs = obs.to(self.config["device"])
                if obs.dim() == 1:
                    obs = obs.unsqueeze(0)
                q_values = self.network_local(obs)
            else:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.config["device"])
                if obs_tensor.dim() == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)
                q_values = self.network_local(obs_tensor)
            if on_training and random.random() < self.cur_agent.get("epsilon", 0.1):
                action = random.randint(0, self.num_phase - 1)
            else:
                action = torch.argmax(q_values, dim=1).item()
        self.network_local.train()
        self.current_phase = action
        return action

    def _compute_critic_loss(self, obs, act, rew, next_obs, done):
        q_current = self.network_local(obs)
        act_indices = act.long()
        if act_indices.dim() == 1:
            act_indices = act_indices.unsqueeze(1)
        q_current = q_current.gather(1, act_indices).squeeze(1)
        with torch.no_grad():
            q_next_target = self.network_target(next_obs)
            q_next_max = q_next_target.max(1)[0]
            q_target = rew + (1 - done.float()) * self.gamma * q_next_max
        return F.mse_loss(q_current, q_target)
