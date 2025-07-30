import torch
import torch.nn as nn
import torch.nn.functional as F
from Layer import MixHopConv


class MixHopLayer(nn.Module):
    def __init__(self, input_dim, dim_per_power, adjacency_powers=[0, 1, 2], add_self_loops=True):
        super(MixHopLayer, self).__init__()
        self.dim_per_power = dim_per_power
        self.mixhop_conv = MixHopConv(
            in_channels=input_dim,
            dim_per_power=dim_per_power,  # Output dimension for each adjacency power
            powers=adjacency_powers,
            add_self_loops=add_self_loops
        )

    def forward(self, x, edge_index):
        # Perform MixHop convolution
        return self.mixhop_conv(x, edge_index)


class MixHopModel(nn.Module):
    def __init__(self, edge_index, input_dim, adjacency_powers, first_layer_dim_per_power, hidden_layer_dims_per_power_list):
        super(MixHopModel, self).__init__()
        self.edge_index = edge_index

        # First MixHop layer with user-defined output dimension per power
        self.mixhop1 = MixHopLayer(input_dim=input_dim, dim_per_power=first_layer_dim_per_power, adjacency_powers=adjacency_powers)

        # Second MixHop layer; input dim is the sum of output dims from the first layer
        self.mixhop2 = MixHopLayer(input_dim=sum(first_layer_dim_per_power), dim_per_power=hidden_layer_dims_per_power_list[0], adjacency_powers=adjacency_powers)

        # Third MixHop layer
        self.mixhop3 = MixHopLayer(input_dim=sum(hidden_layer_dims_per_power_list[0]), dim_per_power=hidden_layer_dims_per_power_list[1], adjacency_powers=adjacency_powers)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        # Pass through three MixHop layers with tanh activation
        x = torch.tanh(self.mixhop1(x, edge_index))
        x = torch.tanh(self.mixhop2(x, edge_index))
        embeddings = torch.tanh(self.mixhop3(x, edge_index))
        return embeddings


class Model(nn.Module):
    def __init__(self, encoder_out: MixHopModel, encoder_in: MixHopModel, num_proj_hidden: int, tau: float):
        super(Model, self).__init__()
        self.encoder_out: MixHopModel = encoder_out
        self.encoder_in: MixHopModel = encoder_in
        self.tau: float = tau

        # Projection head: input dim is the sum of output dims from the third MixHop layer
        encoder_in_dim = sum(encoder_in.mixhop3.dim_per_power)
        encoder_out_dim = sum(encoder_out.mixhop3.dim_per_power)
        self.fc1 = nn.Linear(encoder_out_dim, num_proj_hidden)
        self.fc2 = nn.Linear(num_proj_hidden, encoder_out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple:
        # Encode features using both forward and reversed edge directions
        zout = self.encoder_out(x, edge_index)            # Outward direction embeddings
        zin = self.encoder_in(x, edge_index[[1, 0], :])   # Inward direction embeddings (reversed edges)
        return zout, zin

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        # Non-linear projection head: ELU + Linear
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        # Compute pairwise cosine similarity after L2 normalization
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        # Contrastive loss for self-supervised training
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))         # Positive samples (same view)
        between_sim = f(self.sim(z1, z2))      # Negative samples (different views)

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())
        )

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        # Compute semi-supervised loss in mini-batches to save memory
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))      # [B, N]
            between_sim = f(self.sim(z1[mask], z2))   # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())
            ))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True, batch_size: int = 0):
        # Full loss: symmetric contrastive loss with optional batching
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret
