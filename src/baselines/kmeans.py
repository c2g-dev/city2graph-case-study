from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import torch

def run_kmeans(X, n_clusters, output_path=None, seed=42):
    """
    Apply K-Means clustering to the input features.
    
    Args:
        X: Input features (numpy array or torch Tensor)
        n_clusters: Number of clusters
        output_path: Path to save the cluster labels CSV (optional)
        seed: Random seed
        
    Returns:
        labels: Cluster labels
        kmeans: Fitted KMeans object
    """
    if isinstance(X, torch.Tensor):
        X = X.cpu().detach().numpy()
        
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    labels = kmeans.fit_predict(X)
    
    if output_path:
        df = pd.DataFrame({'cluster': labels})
        df.to_csv(output_path, index=False)
        
    return labels, kmeans
