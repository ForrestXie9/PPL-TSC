from agent import BaseDRL
import torch
import torch.optim as optim
import random
import torch.nn.functional as F
from utilities.utils import copy_model_params
from env.TSC_env import TSCEnv


class AdvancedCoLight(BaseDRL):
    """
    PyTorch 版 Advanced-CoLight。
    在原 CoLight 结构基础上增大 embedding / attention / representation 维度与头数，
    提高网络容量，同时保持与现有环境 / replay_buffer 完全兼容。
    """

    def __init__(self, config, env, idx):
        super(AdvancedCoLight, self).__init__(config, env, idx)

        n_neighbor_idx = env.n_intersection[idx].n_neighbor_idx
        max_num_phase = max(len(inter.n_phase) for inter in env.n_intersection)
        self.network_local = _AdvancedCoLightNetwork(
            num_lane=len(self.inter.n_lane_id),
            num_phase=len(self.inter.n_phase),
            env=env,
            idx=idx,
            n_neighbor_idx=n_neighbor_idx,
            is_target_network=False,
            max_num_phase=max_num_phase,
        )
        self.network_target = _AdvancedCoLightNetwork(
            num_lane=len(self.inter.n_lane_id),
            num_phase=len(self.inter.n_phase),
            env=env,
            idx=idx,
            n_neighbor_idx=n_neighbor_idx,
            is_target_network=True,
            max_num_phase=max_num_phase,
        )

        self.network_local = self.network_local.to(self.config["device"])
        self.network_target = self.network_target.to(self.config["device"])

        self.network_optim = optim.RMSprop(
            self.network_local.parameters(),
            lr=self.cur_agent["learning_rate"],
        )
        copy_model_params(self.network_local, self.network_target)

    def pick_action(self, n_obs, on_training):
        assert all(obs[0].shape[0] == 1 for obs in n_obs), "should be mini-batch with size 1"
        self.network_local.eval()
        with torch.no_grad():
            b_q_value = self.network_local(n_obs)
            q_valid = b_q_value[:, : self.num_phase]
            action = torch.argmax(q_valid, dim=1).item()
            if on_training and random.random() < self.cur_agent["epsilon"]:
                action = random.randint(0, self.num_phase - 1)
        self.network_local.train()
        self.current_phase = action
        return self.current_phase

    def learn(self):
        if self._time_to_learn():
            idxs = self.replay_buffer.get_sample_indexes()
            _, act, rew, _, done = self.replay_buffer.sample_experience(idxs=idxs)
            n_obs, n_next_obs = [], []
            for agent in self.env.n_agent:
                obs, _, _, next_obs, _ = agent.replay_buffer.sample_experience(idxs=idxs)
                n_obs.append(obs)
                n_next_obs.append(next_obs)

            critic_loss = self._compute_critic_loss(n_obs, act, rew, n_next_obs, done)
            self.network_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network_local.parameters(), max_norm=10.0)
            self.network_optim.step()
            for to_model, from_model in zip(self.network_target.parameters(), self.network_local.parameters()):
                to_model.data.copy_(self.tau * from_model.data + (1.0 - self.tau) * to_model.data)
            return critic_loss
        else:
            return 0

    def _compute_critic_loss(self, n_obs, act, rew, n_next_obs, done):
        with torch.no_grad():
            q_target_next = self.network_target(n_next_obs)
            q_target_next_valid = q_target_next[:, : self.num_phase]
            q_target = rew + self.gamma * torch.max(q_target_next_valid, dim=1, keepdim=True)[0] * (~done)
        q_expected_full = self.network_local(n_obs)
        q_expected_valid = q_expected_full[:, : self.num_phase]
        q_expected = q_expected_valid.gather(1, act.long())
        critic_loss = F.mse_loss(q_expected, q_target)
        return critic_loss


class _AdvancedCoLightNetwork(torch.nn.Module):
    """
    加宽版 CoLight 网络：增大 embedding / attention head / representation 以及 head 数量。
    """

    def __init__(self, num_lane, num_phase, env: TSCEnv, idx, n_neighbor_idx, is_target_network, max_num_phase=None):
        super(_AdvancedCoLightNetwork, self).__init__()
        self.num_lane = num_lane
        self.num_phase = num_phase
        self.env = env
        self.idx = idx
        self.n_neighbor_idx = n_neighbor_idx
        self.is_target_network = is_target_network
        self.max_num_phase = max_num_phase if max_num_phase is not None else num_phase

        # 相比原 CoLight：通道数翻倍、head 数增加；Advanced 版输入为 phase + num_vehicle + delay，共 3 路
        self.dim_embedding = 64
        self.dim_attention_head = 64
        self.dim_representation = 64
        self.num_head = 8
        # AdvancedCoLight: inter_2_current_phase + lane_2_num_vehicle + lane_2_delay
        self.input_dim = self.num_phase + self.num_lane * 2

        if idx == 0:
            self.embedding_mlp = torch.nn.Sequential(
                torch.nn.Linear(in_features=self.input_dim, out_features=self.dim_embedding),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=self.dim_embedding, out_features=self.dim_embedding),
                torch.nn.ReLU(),
            )

            self.n_agent_embedding_2_head = torch.nn.ModuleList(
                [
                    torch.nn.Sequential(
                        torch.nn.Linear(in_features=self.dim_embedding, out_features=self.dim_attention_head),
                        torch.nn.ReLU(),
                    )
                    for _ in range(self.num_head)
                ]
            )

            self.n_neighbor_embedding_2_head = torch.nn.ModuleList(
                [
                    torch.nn.Sequential(
                        torch.nn.Linear(in_features=self.dim_embedding, out_features=self.dim_attention_head),
                        torch.nn.ReLU(),
                    )
                    for _ in range(self.num_head)
                ]
            )

            self.n_neighbor_embedding_2_hidden_repr = torch.nn.ModuleList(
                [
                    torch.nn.Sequential(
                        torch.nn.Linear(in_features=self.dim_embedding, out_features=self.dim_representation),
                        torch.nn.ReLU(),
                    )
                    for _ in range(self.num_head)
                ]
            )

            self.tailing_net = torch.nn.Sequential(
                torch.nn.Linear(in_features=self.dim_representation, out_features=self.dim_representation),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=self.dim_representation, out_features=self.max_num_phase),
            )
        elif self.is_target_network:
            self.embedding_mlp = self.env.n_agent[0].network_target.embedding_mlp
            self.n_agent_embedding_2_head = self.env.n_agent[0].network_target.n_agent_embedding_2_head
            self.n_neighbor_embedding_2_head = self.env.n_agent[0].network_target.n_neighbor_embedding_2_head
            self.n_neighbor_embedding_2_hidden_repr = (
                self.env.n_agent[0].network_target.n_neighbor_embedding_2_hidden_repr
            )
            self.tailing_net = self.env.n_agent[0].network_target.tailing_net
        else:
            self.embedding_mlp = self.env.n_agent[0].network_local.embedding_mlp
            self.n_agent_embedding_2_head = self.env.n_agent[0].network_local.n_agent_embedding_2_head
            self.n_neighbor_embedding_2_head = self.env.n_agent[0].network_local.n_neighbor_embedding_2_head
            self.n_neighbor_embedding_2_hidden_repr = (
                self.env.n_agent[0].network_local.n_neighbor_embedding_2_hidden_repr
            )
            self.tailing_net = self.env.n_agent[0].network_local.tailing_net

    def forward(self, n_obs):
        n_neighbor_idx = self.env.n_intersection[self.idx].n_neighbor_idx
        expected_dim = self.embedding_mlp[0].in_features

        def _pad_obs(obs_list, target_dim):
            flat = torch.cat(obs_list, dim=1)
            if flat.shape[1] < target_dim:
                pad = torch.zeros(flat.shape[0], target_dim - flat.shape[1], device=flat.device, dtype=flat.dtype)
                flat = torch.cat([flat, pad], dim=1)
            elif flat.shape[1] > target_dim:
                flat = flat[:, :target_dim]
            return flat

        all_flatten_feature = _pad_obs(n_obs[self.idx], expected_dim)
        agent_embedding = self.embedding_mlp(all_flatten_feature)

        n_neighbor_embedding = torch.stack(
            [
                self.env.n_agent[0].network_local.embedding_mlp(_pad_obs(n_obs[neighbor_idx], expected_dim))
                for neighbor_idx in n_neighbor_idx
            ],
            dim=1,
        )

        n_agent_hidden_repr = []
        for head_idx in range(self.num_head):
            agent_attention_head = self.n_agent_embedding_2_head[head_idx](agent_embedding).unsqueeze(2)
            n_neighbor_attention_head = self.n_neighbor_embedding_2_head[head_idx](n_neighbor_embedding)

            attention = torch.softmax(torch.bmm(n_neighbor_attention_head, agent_attention_head), dim=1)
            attention = attention.permute(0, 2, 1)

            n_neighbor_hidden_repr = self.n_neighbor_embedding_2_hidden_repr[head_idx](n_neighbor_embedding)
            agent_hidden_repr = torch.bmm(attention, n_neighbor_hidden_repr)
            agent_hidden_repr = agent_hidden_repr.squeeze(1)
            n_agent_hidden_repr.append(agent_hidden_repr)

        avg_agent_hidden_repr = torch.mean(torch.stack(n_agent_hidden_repr, dim=2), dim=2)
        q_values = self.tailing_net(avg_agent_hidden_repr)
        return q_values

