import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from .utils import structure_loss_distmult

class HANLayer(nn.Module):
    def __init__(self, in_dim, out_dim, metapaths, heads=4, dropout=0.3):
        super().__init__()
        self.metapaths = metapaths
        self.dropout = dropout

        # 1. Node-Level Attention: One GAT per meta-path
        # Each meta-path gets its own set of parameters to learn unique features
        self.gat_layers = nn.ModuleDict()
        for mp in metapaths:
            self.gat_layers[mp] = GATConv(in_dim, out_dim, heads=heads,
                                          concat=False, dropout=dropout)

        # 2. Semantic-Level Attention Components
        # Projects all meta-path embeddings to a common space to measure importance
        self.semantic_proj = nn.Linear(out_dim, out_dim)
        self.semantic_attn_vec = nn.Parameter(torch.Tensor(1, out_dim))
        nn.init.xavier_uniform_(self.semantic_attn_vec)

    def forward(self, x, hetero_data):
        semantic_embeddings = []

        # Step 1: Generate embedding for each meta-path
        for mp in self.metapaths:
            # Extract specific edge index (e.g., transit_15)
            edge_index = hetero_data[('oa', mp, 'oa')].edge_index

            # Apply GAT
            z_mp = self.gat_layers[mp](x, edge_index)
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
    def __init__(self, in_dim, hidden_dim, out_dim, metapaths, heads=4, dropout=0.3):
        super().__init__()
        self.metapaths = metapaths

        # Encoder: 2-Layer HAN
        # Layer 1: Expands features, learning high-level concepts
        self.han1 = HANLayer(in_dim, hidden_dim, metapaths, heads, dropout)
        # Layer 2: Compresses features, refining the latent space
        self.han2 = HANLayer(hidden_dim, out_dim, metapaths, heads, dropout)

        # Feature Decoder
        self.feat_decoder = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.ELU(),
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

    def encode(self, x, hetero_data):
        # Layer 1
        h, _ = self.han1(x, hetero_data)
        h = F.dropout(h, p=0.3, training=self.training)

        # Layer 2 (We extract beta here for explainability analysis)
        z, beta = self.han2(h, hetero_data)
        return z, beta

    def forward(self, hetero_data):
        z, beta = self.encode(hetero_data['oa'].x, hetero_data)
        z = F.normalize(z, p=2, dim=-1)
        x_hat = self.feat_decoder(z)
        return z, x_hat, beta

    def compute_loss(self, data, lambda_struct=1.0, neg_sampling_scale=1.0):
        z, x_hat, _ = self.forward(data)

        # 1. Feature Loss
        l_feat = F.smooth_l1_loss(x_hat, data['oa'].x)

        # 2. Structure Loss (Averaged over all meta-paths)
        num_nodes = data['oa'].num_nodes
        l_struct_total = 0
        for mp in self.metapaths:
            edge_index = data[('oa', mp, 'oa')].edge_index
            
            # Calculate dynamic negative sampling ratio based on density
            num_edges = edge_index.size(1)
            density = num_edges / (num_nodes ** 2)
            # Avoid division by zero
            ratio = neg_sampling_scale / density

            l_struct_total += structure_loss_distmult(
                z, edge_index, self.struct_decoders[mp], neg_sampling_ratio=ratio
            )
        
        # Standardize loss to be independent of number of metapaths
        l_struct_mean = l_struct_total / len(self.metapaths)

        return l_feat + lambda_struct * l_struct_mean, l_feat, l_struct_mean
