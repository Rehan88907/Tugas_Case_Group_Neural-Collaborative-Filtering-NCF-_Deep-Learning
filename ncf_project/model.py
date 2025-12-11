import torch
import torch.nn as nn

class NeuMF(nn.Module):
    def __init__(self, n_users, n_items, gmf_dim=8, mlp_dim=16):
        super().__init__()

        # Embedding GMF
        self.gmf_user = nn.Embedding(n_users, gmf_dim)
        self.gmf_item = nn.Embedding(n_items, gmf_dim)

        # Embedding MLP
        self.mlp_user = nn.Embedding(n_users, mlp_dim)
        self.mlp_item = nn.Embedding(n_items, mlp_dim)

        self.mlp_layers = nn.Sequential(
            nn.Linear(mlp_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.output = nn.Linear(gmf_dim + 32, 1)

    def forward(self, user, item):
        gmf_u = self.gmf_user(user)
        gmf_i = self.gmf_item(item)
        gmf_out = gmf_u * gmf_i

        mlp_u = self.mlp_user(user)
        mlp_i = self.mlp_item(item)
        mlp_cat = torch.cat([mlp_u, mlp_i], dim=-1)
        mlp_out = self.mlp_layers(mlp_cat)

        final = torch.cat([gmf_out, mlp_out], dim=-1)
        return torch.sigmoid(self.output(final)).squeeze()
