import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv

from .utils import structure_loss_distmult

class HANLayer(nn.Module):
    """
    Hierarchical Attention Network (HAN) Layer.
    
    Performs node-level attention (using GAT) for each meta-path, followed by 
    semantic-level attention to aggregate features from different meta-paths.
    """
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        metapaths: list[str], 
        heads: int = 4, 
        dropout: float = 0.3, 
        edge_dim_dict: dict[str, int] | None = None
    ):
        super().__init__()
        self.metapaths = metapaths
        self.dropout = dropout
        self.edge_dim_dict = edge_dim_dict or {}

        # 1. Node-Level Attention: One GAT per meta-path
        self.gat_layers = nn.ModuleDict()
        for mp in metapaths:
            edge_dim = self.edge_dim_dict.get(mp, None)
            # Note: concat=False reduces output dimension to `out_dim` (averaging heads),
            # which simplifies semantic attention projection.
            self.gat_layers[mp] = GATConv(
                in_dim, out_dim, heads=heads,
                concat=False, dropout=dropout, edge_dim=edge_dim
            )

        # Edge Normalization: One BN per meta-path
        self.edge_norm_layers = nn.ModuleDict()
        for mp in metapaths:
            # Default to 1 if edge_dim is not specified, preventing KeyErrors
            dim = self.edge_dim_dict.get(mp, 1)
            self.edge_norm_layers[mp] = nn.BatchNorm1d(dim)

        # 2. Semantic-Level Attention Components
        # Projects all meta-path embeddings to a common space to measure importance
        self.semantic_proj = nn.Linear(out_dim, out_dim)
        self.semantic_attn_vec = nn.Parameter(torch.Tensor(1, out_dim))
        nn.init.xavier_uniform_(self.semantic_attn_vec)

    def forward(self, x: torch.Tensor, hetero_data: HeteroData) -> tuple[torch.Tensor, torch.Tensor]:
        semantic_embeddings = []

        # Step 1: Generate embedding for each meta-path
        for mp in self.metapaths:
            # Extract specific edge index
            edge_store = hetero_data[('oa', mp, 'oa')]
            edge_index = edge_store.edge_index
            
            # Log-transform and normalize edge attributes
            # Check if edge_attr exists to avoid errors, although HAN expects them if edge_dim passed
            if hasattr(edge_store, 'edge_attr') and edge_store.edge_attr is not None:
                edge_attr = self.edge_norm_layers[mp](torch.log1p(edge_store.edge_attr))
            else:
                edge_attr = None

            # Apply GAT
            z_mp = self.gat_layers[mp](x, edge_index, edge_attr=edge_attr)
            z_mp = F.elu(z_mp)
            semantic_embeddings.append(z_mp)

        # Stack: [N, Num_Metapaths, Out_Dim]
        z_stack = torch.stack(semantic_embeddings, dim=1)

        # Step 2: Semantic Attention Mechanism
        # Non-linear transformation
        weights = torch.tanh(self.semantic_proj(z_stack))
        
        # Similarity with learnable attention vector q
        scores = (weights * self.semantic_attn_vec).sum(dim=-1) # [N, Num_Metapaths]
        
        # Beta: The normalized importance of each transport mode per node
        beta = F.softmax(scores, dim=1)

        # Weighted Sum Fusion
        z_final = (z_stack * beta.unsqueeze(-1)).sum(dim=1)

        return z_final, beta

class HANGAE(nn.Module):
    """
    HAN-based Graph Autoencoder (HANGAE).
    
    Encoder: 2-Layer HAN (Hierarchical Attention Network).
    Decoder: 
        - Feature Decoder: MLP
        - Structure Decoder: DistMult (one per meta-path)
    """
    def __init__(
        self, 
        in_dim: int, 
        hidden_dim: int, 
        out_dim: int, 
        metapaths: list[str], 
        heads: int = 4, 
        dropout: float = 0.3, 
        edge_dim_dict: dict[str, int] | None = None
    ):
        super().__init__()
        self.metapaths = metapaths
        self.dropout = dropout

        # Encoder: 2-Layer HAN
        # Layer 1: Expands features, learning high-level concepts
        self.han1 = HANLayer(in_dim, hidden_dim, metapaths, heads, dropout, edge_dim_dict)
        
        # Layer 2: Compresses features, refining the latent space
        self.han2 = HANLayer(hidden_dim, out_dim, metapaths, heads=1, dropout=dropout, edge_dim_dict=edge_dim_dict)

        # Feature Decoder
        self.feat_decoder = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim)
        )

        # Structure Decoders (One per meta-path)
        # We need separate DistMult vectors because 'connectivity' means different
        # things in different graphs (e.g., bus route vs. street)
        self.struct_decoders = nn.ParameterDict({
            mp: nn.Parameter(torch.Tensor(out_dim)) for mp in metapaths
        })
        for param in self.struct_decoders.values():
            nn.init.xavier_uniform_(param.unsqueeze(0))

    def encode(self, x: torch.Tensor, hetero_data: HeteroData) -> tuple[torch.Tensor, torch.Tensor]:
        # Layer 1
        h, _ = self.han1(x, hetero_data)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Layer 2 (Extract beta for explainability)
        z, beta = self.han2(h, hetero_data)
        return z, beta

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes the latent representation back to the original feature space."""
        return self.feat_decoder(z)

    def forward(self, hetero_data: HeteroData) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, beta = self.encode(hetero_data['oa'].x, hetero_data)
        x_hat = self.decode(z)
        return z, x_hat, beta

    def compute_loss(
        self, 
        data: HeteroData, 
        lambda_struct: float = 1.0, 
        neg_sampling_ratio: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, x_hat, _ = self.forward(data)

        # 1. Feature Loss
        l_feat = F.smooth_l1_loss(x_hat, data['oa'].x)

        # 2. Structure Loss (Averaged over all meta-paths)
        l_struct_total = 0
        for mp in self.metapaths:
            edge_index = data[('oa', mp, 'oa')].edge_index
            
            l_struct_total += structure_loss_distmult(
                z, edge_index, self.struct_decoders[mp], neg_sampling_ratio=neg_sampling_ratio
            )
        
        # Standardize loss to be independent of number of metapaths
        l_struct_mean = l_struct_total / len(self.metapaths)

        return l_feat + lambda_struct * l_struct_mean, l_feat, l_struct_mean
