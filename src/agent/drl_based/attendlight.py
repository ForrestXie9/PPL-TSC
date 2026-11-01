"""AttendLight adapter inherited from LLMTSCS-main.

Uses incoming-lane observations, within-phase attention and attention across phases."""

from agent.drl_based.base_drl import BaseDRL
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


def _build_phase_map_from_intersection(inter):
    """Map each physical phase to its passable incoming-lane indices."""

    phase_map = []
    for phase_idx in range(len(inter.phase_2_passable_lane_idx)):
        lane_indices = [
            j for j in range(len(inter.phase_2_passable_lane_idx[phase_idx]))
            if inter.phase_2_passable_lane_idx[phase_idx][j] == 1
        ]
        phase_map.append(lane_indices)
    return phase_map


class AttendLightNetwork(nn.Module):
    """Embed lane features, attend within each phase, then attend across phase representations."""

    def __init__(self, num_lane, num_phase, phase_map, embed_dim=32, num_heads=4, hidden_dim=20):
        super(AttendLightNetwork, self).__init__()
        self.num_lane = num_lane
        self.num_phase = num_phase
        self.phase_map = phase_map  # Index metadata only; phase_map is not a learned parameter.
        self.embed_dim = embed_dim
        if embed_dim % num_heads != 0:
            embed_dim = ((embed_dim + num_heads - 1) // num_heads) * num_heads
        self.embed_dim = embed_dim


        self.embed = nn.Linear(1, embed_dim)

        # Attend within each phase using the mean lane representation as the query.
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



        print(f"[AttendLightNetwork] num_lane={num_lane}, num_phase={num_phase}, embed_dim={embed_dim}, num_heads={num_heads}")

    def forward(self, x):
        """Accept a (batch, num_lane) tensor or a list of tensors concatenated along the feature axis."""
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

        # Build one (B, 1, embed_dim) representation per phase.
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
    """AttendLight adapter using incoming-lane features and two attention stages."""

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
        # Enforce minimum green only after the phase clock has started.
        # At time zero, allow a Q-value decision rather than locking phase zero.
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
