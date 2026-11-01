from agent import BaseDRL
import torch
import torch.optim as optim
from utilities.utils import copy_model_params


class AdvancedMPLight(BaseDRL):
    """Optional wider PyTorch MPLight variant with the existing environment and replay interfaces."""

    def __init__(self, config, env, idx):
        super(AdvancedMPLight, self).__init__(config, env, idx)
        self.phase_2_passable_lanelink = torch.tensor(self.inter.phase_2_passable_lanelink_idx)
        self.network_local = _AdvancedMPLightNetwork(
            num_lanelink=sum(self.inter.n_num_lanelink),
            num_phase=len(self.inter.n_phase),
            phase_2_passable_lanelink=self.phase_2_passable_lanelink,
        )
        self.network_target = _AdvancedMPLightNetwork(
            num_lanelink=sum(self.inter.n_num_lanelink),
            num_phase=len(self.inter.n_phase),
            phase_2_passable_lanelink=self.phase_2_passable_lanelink,
        )

        # Move models to device
        self.network_local = self.network_local.to(self.config["device"])
        self.network_target = self.network_target.to(self.config["device"])
        self.phase_2_passable_lanelink = self.phase_2_passable_lanelink.to(self.config["device"])

        self.network_optim = optim.RMSprop(
            self.network_local.parameters(),
            lr=self.cur_agent["learning_rate"],
        )
        copy_model_params(self.network_local, self.network_target)


class _AdvancedMPLightNetwork(torch.nn.Module):
    """Wider FRAP-style network with unchanged input and output interfaces."""

    def __init__(self, num_lanelink, num_phase, phase_2_passable_lanelink):
        super(_AdvancedMPLightNetwork, self).__init__()
        self.num_lanelink = num_lanelink
        self.num_phase = num_phase
        self.phase_2_passable_lanelink = phase_2_passable_lanelink
        self.lanelink_2_applicable_phase = phase_2_passable_lanelink.permute(1, 0)  # num_lanelink * num_phase
        self.phase_competition_mask = self._get_phase_competition_mask(phase_2_passable_lanelink)

        # Embed phase availability, lane-link pressure and lane-link demand.
        self.dim_embedding = 4
        self.dim_hidden_repr = 24
        self.dim_conv_repr = 32

        # Binary passability under the current phase.
        self.phase_embedding = torch.nn.Sequential(
            torch.nn.Linear(in_features=1, out_features=self.dim_embedding),
            torch.nn.ReLU(),
        )
        # Lane-link pressure feature.
        self.pressure_embedding = torch.nn.Sequential(
            torch.nn.Linear(in_features=1, out_features=self.dim_embedding),
            torch.nn.ReLU(),
        )
        # Lane-link vehicle-demand feature.
        self.demand_embedding = torch.nn.Sequential(
            torch.nn.Linear(in_features=1, out_features=self.dim_embedding),
            torch.nn.ReLU(),
        )
        # Map the concatenated embeddings to lane-link representations.
        self.lanelink_embedding = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.dim_embedding * 3, out_features=self.dim_hidden_repr),
            torch.nn.ReLU(),
        )

        self.relation_embedding = torch.nn.Embedding(num_embeddings=2, embedding_dim=self.dim_embedding)

        self.conv_cube = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=self.dim_hidden_repr * 2,
                out_channels=self.dim_conv_repr,
                kernel_size=(1, 1),
            ),
            torch.nn.ReLU(),
        )
        self.conv_relation = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=self.dim_embedding,
                out_channels=self.dim_conv_repr,
                kernel_size=(1, 1),
            ),
            torch.nn.ReLU(),
        )
        self.tail_layer = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=self.dim_conv_repr,
                out_channels=self.dim_conv_repr,
                kernel_size=(1, 1),
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=self.dim_conv_repr,
                out_channels=1,
                kernel_size=(1, 1),
            ),
        )

    def forward(self, obs):
        """
        obs[0]: inter_2_current_phase (one-hot), shape [B, num_phase]
        obs[1]: lanelink_2_pressure,            shape [B, num_lanelink]
        obs[2]: lanelink_2_num_vehicle,         shape [B, num_lanelink]
        """
        batch_size = obs[0].shape[0]
        b_current_passable_lanelink = torch.matmul(
            obs[0].float(), self.phase_2_passable_lanelink.float()
        )  # batch * num_lanelink
        b_pressure = obs[1]
        b_demand = obs[2]

        # Embed phase availability, pressure and demand.
        lanelink_phase_emb = self.phase_embedding(
            b_current_passable_lanelink.unsqueeze(-1)
        )  # batch * num_lanelink * dim_embedding
        lanelink_pressure_emb = self.pressure_embedding(b_pressure.unsqueeze(-1))
        lanelink_demand_emb = self.demand_embedding(b_demand.unsqueeze(-1))

        lanelink_embedding = torch.cat(
            [lanelink_phase_emb, lanelink_pressure_emb, lanelink_demand_emb], dim=2
        )  # batch * num_lanelink * (3 * dim_embedding)
        lanelink_embedding = self.lanelink_embedding(lanelink_embedding)
        lanelink_embedding = lanelink_embedding.permute(0, 2, 1)  # batch * dim_hidden_repr * num_lanelink

        phase_embedding = torch.matmul(
            lanelink_embedding, self.lanelink_2_applicable_phase.float()
        )  # batch * dim_hidden_repr * num_phase
        phase_embedding = phase_embedding.permute(0, 2, 1)  # batch * num_phase * dim_hidden_repr
        phase_embedding_cube = self._get_phase_embedding_cube(batch_size, phase_embedding)

        phase_conv = self.conv_cube(phase_embedding_cube)

        relation_embedding = self.relation_embedding(self.phase_competition_mask).permute(
            2, 0, 1
        ).unsqueeze(0)
        relation_conv = self.conv_relation(relation_embedding)

        combined_feature = phase_conv * relation_conv
        before_merge = self.tail_layer(combined_feature)
        q_values = torch.sum(before_merge, dim=3).squeeze(1)
        return q_values

    def _get_phase_embedding_cube(self, batch_size, phase_embedding):
        # Use dim_hidden_repr * 2 instead of FRAP's fixed width of 32.
        cube_channels = self.dim_hidden_repr * 2
        phase_embedding_cube = torch.zeros(
            (batch_size, cube_channels, self.num_phase, self.num_phase - 1),
            device=phase_embedding.device,
            dtype=phase_embedding.dtype,
        )
        for phase_idx in range(self.num_phase):
            continuous_jdx = 0
            for phase_jdx in range(self.num_phase):
                if phase_idx == phase_jdx:
                    continue
                phase_embedding_comb = torch.cat(
                    [
                        phase_embedding[:, phase_idx, :],
                        phase_embedding[:, phase_jdx, :],
                    ],
                    dim=1,
                )
                phase_embedding_cube[:, :, phase_idx, continuous_jdx] = phase_embedding_comb
                continuous_jdx += 1
        return phase_embedding_cube

    def _get_phase_competition_mask(self, phase_2_passable_lanelink):
        mask = torch.zeros((self.num_phase, self.num_phase - 1), dtype=torch.int64)
        for phase_idx in range(self.num_phase):
            continuous_jdx = 0
            for phase_jdx in range(self.num_phase):
                if phase_idx == phase_jdx:
                    continue
                for lanelink_idx in range(self.num_lanelink):
                    if (
                        phase_2_passable_lanelink[phase_idx][lanelink_idx]
                        == phase_2_passable_lanelink[phase_jdx][lanelink_idx]
                        == 1
                    ):
                        mask[phase_idx][continuous_jdx] = 1
                continuous_jdx += 1
        return mask

