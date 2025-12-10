import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

from .utils import structure_loss_distmult

class GATGAE(nn.Module):
    """
    GAT-based Graph Autoencoder (GATGAE).
    
    Encoder: 2-layer GAT.
    Decoder: 
        - Feature Decoder: MLP
        - Structure Decoder: DistMult (single relation)
    """
    def __init__(
        self, 
        in_dim: int, 
        hidden_dim: int, 
        out_dim: int, 
        heads: int = 4, 
        dropout: float = 0.6, 
        edge_dim: int = 1,
        use_features: bool = True,
        num_nodes: int = 0
    ):
        super().__init__()
        self.dropout = dropout
        self.use_features = use_features
        self.in_dim = in_dim

        # Encoder: 2-layer GAT
        # Layer 1: Learns local feature aggregation
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, concat=True, dropout=dropout, edge_dim=edge_dim)

        # Layer 2: Compresses to latent dimension 'out_dim'
        self.conv2 = GATConv(hidden_dim * heads, out_dim, heads=1, concat=True, dropout=dropout, edge_dim=edge_dim)

        # Feature Decoder: MLP (only needed when using features)
        if use_features:
            self.feat_decoder = nn.Sequential(
                nn.Linear(out_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, in_dim)
            )

        # Structure Decoder: Learnable Diagonal for DistMult
        # A single relation vector because this model only sees 'contiguity'
        self.struct_decoder_rel = nn.Parameter(torch.Tensor(out_dim))
        nn.init.xavier_uniform_(self.struct_decoder_rel.unsqueeze(0))

        # Edge Normalization
        self.edge_norm = nn.BatchNorm1d(edge_dim)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # 1. Edge Feature Preprocessing
        # Log-transform and normalize for numeric stability
        edge_attr = self.edge_norm(torch.log1p(edge_attr))

        # 2. GAT Encoder
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        # Keep one explicit feature-dropout point between encoder layers.
        x = F.dropout(x, p=self.dropout, training=self.training)
        z = self.conv2(x, edge_index, edge_attr=edge_attr)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor | None:
        """Decodes the latent representation back to the original feature space."""
        if self.use_features:
            return self.feat_decoder(z)
        return None

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.use_features:
            x = data.x
        else:
            x = torch.ones(data.num_nodes, self.in_dim, device=data.x.device)
        z = self.encode(x, data.edge_index, edge_attr=data.edge_attr)
        x_hat = self.decode(z)
        return z, x_hat

    def compute_loss(
        self, 
        data: Data, 
        lambda_struct: float = 1.0, 
        neg_sampling_ratio: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, x_hat = self.forward(data)

        # 1. Feature Loss (SmoothL1) — skipped in structure-only mode
        if self.use_features:
            l_feat = F.smooth_l1_loss(x_hat, data.x)
        else:
            l_feat = torch.tensor(0.0, device=z.device)

        # 2. Structure Loss (BCE with Negative Sampling)
        l_struct = structure_loss_distmult(z, data.edge_index, self.struct_decoder_rel, neg_sampling_ratio=neg_sampling_ratio)

        if self.use_features:
            return l_feat + lambda_struct * l_struct, l_feat, l_struct
        else:
            return l_struct, l_feat, l_struct
