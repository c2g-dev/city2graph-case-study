import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from .utils import structure_loss_distmult

class GATGAE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=4, dropout=0.3):
        super().__init__()
        self.dropout = dropout

        # Encoder: 2-layer GAT
        # Layer 1: Learns local feature aggregation
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout)
        # Layer 2: Compresses to latent dimension 'out_dim'
        self.conv2 = GATConv(hidden_dim * heads, out_dim, heads=1, concat=False, dropout=dropout)

        # Feature Decoder: MLP
        self.feat_decoder = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, in_dim)
        )

        # Structure Decoder: Learnable Diagonal for DistMult
        # A single relation vector because this model only sees 'contiguity'
        self.struct_decoder_rel = nn.Parameter(torch.Tensor(out_dim))
        nn.init.xavier_uniform_(self.struct_decoder_rel.unsqueeze(0))

    def encode(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        z = self.conv2(x, edge_index)
        return z

    def forward(self, data):
        # Model 1 uses only 'is_contiguous_to' edge_index
        z = self.encode(data.x, data.edge_index)
        z = F.normalize(z, p=2, dim=-1)
        x_hat = self.feat_decoder(z)
        return z, x_hat

    def compute_loss(self, data, lambda_struct=1.0, neg_sampling_scale=1.0):
        z, x_hat = self.forward(data)

        # 1. Feature Loss (SmoothL1)
        l_feat = F.smooth_l1_loss(x_hat, data.x)

        # 2. Structure Loss (BCE with Negative Sampling)
        # Calculate dynamic negative sampling ratio based on density
        num_nodes = data.x.size(0)
        num_edges = data.edge_index.size(1)
        density = num_edges / (num_nodes ** 2)
        ratio = neg_sampling_scale / density

        l_struct = structure_loss_distmult(z, data.edge_index, self.struct_decoder_rel, neg_sampling_ratio=ratio)

        return l_feat + lambda_struct * l_struct, l_feat, l_struct
