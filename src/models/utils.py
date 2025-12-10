import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling

def structure_loss_distmult(z, edge_index, relation_vector, neg_sampling_ratio=1.0):
    """
    Computes BCE loss using DistMult decoder.
    DistMult Score function: s(u,v) = z_u * R * z_v

    Args:
        z: Node embeddings [N, d]
        edge_index: Graph connectivity [2, E]
        relation_vector: Learnable diagonal matrix R (vector form) [d]
        neg_sampling_ratio: Ratio of negative samples to positive samples (default: 1.0)
    """
    # 1. Positive Samples (Actual links in the graph)
    src, dst = edge_index
    # Element-wise multiplication represents diagonal matrix multiplication
    score_pos = (z[src] * relation_vector * z[dst]).sum(dim=-1)

    # 2. Negative Samples (Non-existent links)
    num_neg_samples = int(edge_index.size(1) * neg_sampling_ratio)
    neg_src, neg_dst = negative_sampling(edge_index, num_nodes=z.size(0), num_neg_samples=num_neg_samples)
    score_neg = (z[neg_src] * relation_vector * z[neg_dst]).sum(dim=-1)

    # 3. Binary Cross Entropy with Logits
    # We want score_pos -> 1 (High probability of link)
    loss_pos = F.binary_cross_entropy_with_logits(score_pos, torch.ones_like(score_pos))
    # We want score_neg -> 0 (Low probability of link)
    loss_neg = F.binary_cross_entropy_with_logits(score_neg, torch.zeros_like(score_neg))

    return loss_pos + loss_neg
