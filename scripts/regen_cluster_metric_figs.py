"""Regenerate the Figure 12 / Appendix D cluster-metric panels in selectable styles.

This reproduces the data setup of notebooks/04_evaluation.ipynb (cells 3, 5, 7),
caches the per-k clustering search to a pickle so re-rendering is instant, and then
renders the metric panels in one of several plotting styles.

Styles:
  bold      : the revised-manuscript style. Structure-only variants (1S--4S) are
              drawn at full opacity as thicker dashed lines with larger open
              (white-filled) markers, so their point symbols are legible while
              staying distinct from the solid-line proposed models. All nine
              models stay in a single panel per metric (no faceting).
  v1        : the originally submitted style (structure-only = dashed, alpha=0.3).
  v1_solid  : exactly v1 but structure-only lines changed from dashed to solid.

Run from the repo root (city2graph-case-study/):
    .venv/bin/python scripts/regen_cluster_metric_figs.py --style bold
"""
import argparse
import os
import pickle
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath('.'))

import yaml
import torch
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import normalize
import city2graph as c2g

with open('configs/experiment_config.yaml') as f:
    config = yaml.safe_load(f)
np.random.seed(config['seeds']['global'])
torch.manual_seed(config['seeds']['global'])

DATA_ROOT = config['data']['root']
EMBEDDINGS_DIR = 'data/outputs/embeddings'
FIGURES_DIR = 'data/outputs/figures'
CACHE = 'data/outputs/clusters/k_search_cache.pkl'

MODELS = {
    'Baseline (PCA)': {'file': f"{DATA_ROOT}/{config['data']['homo']}", 'type': 'baseline'},
    'Model 1 (GAT-GAE)': {'file': f"{EMBEDDINGS_DIR}/model1_gat_gae.pt", 'type': 'homo'},
    'Model 2 (HAN-GAE: Walk)': {'file': f"{EMBEDDINGS_DIR}/model2_han_gae_walk.pt", 'type': 'hetero'},
    'Model 3 (HAN-GAE: Multi)': {'file': f"{EMBEDDINGS_DIR}/model3_han_gae_multi.pt", 'type': 'hetero'},
    'Model 4 (HAN-GAE: All)': {'file': f"{EMBEDDINGS_DIR}/model4_han_gae_all.pt", 'type': 'hetero'},
    'Model 1S (GAT-GAE: Struct)': {'file': f"{EMBEDDINGS_DIR}/model1s_gat_gae_struct.pt", 'type': 'homo_struct'},
    'Model 2S (HAN-GAE: Walk-S)': {'file': f"{EMBEDDINGS_DIR}/model2s_han_gae_walk_struct.pt", 'type': 'hetero_struct'},
    'Model 3S (HAN-GAE: Multi-S)': {'file': f"{EMBEDDINGS_DIR}/model3s_han_gae_multi_struct.pt", 'type': 'hetero_struct'},
    'Model 4S (HAN-GAE: All-S)': {'file': f"{EMBEDDINGS_DIR}/model4s_han_gae_all_struct.pt", 'type': 'hetero_struct'},
}
EMBEDDING_COLS = [f'embedding_{i}' for i in range(config['model']['out_dim'])]
K_RANGE = range(2, 51)


def compute_clustering():
    oa_polygons = gpd.read_file('data/processed/features/oa_with_features.gpkg').set_index('OA21CD')
    model_gdfs = {}
    for model_name, meta in MODELS.items():
        data = torch.load(meta['file'], map_location='cpu', weights_only=False)
        if meta['type'] == 'baseline':
            nodes_gdf, _ = c2g.pyg_to_gdf(data, node_types='oa')
        elif meta['type'].startswith('hetero'):
            nodes_gdf = c2g.pyg_to_gdf(data, node_types=['oa'], additional_node_cols={'oa': EMBEDDING_COLS})[0]['oa']
        else:
            nodes_gdf, _ = c2g.pyg_to_gdf(data, additional_node_cols=EMBEDDING_COLS)
        model_gdfs[model_name] = nodes_gdf.set_geometry(oa_polygons.loc[nodes_gdf.index, 'geometry'])

    data_multi = torch.load(f"{DATA_ROOT}/{config['data']['hetero_multi']}", map_location='cpu', weights_only=False)
    oa_edge_types = [('oa', 'contig', 'oa'), ('oa', '15_min_walk', 'oa'), ('oa', '15_min_multi', 'oa')]
    oa_nodes_multi, oa_edges_multi = c2g.pyg_to_gdf(data_multi, node_types=['oa'], edge_types=oa_edge_types)
    oa_nodes_gdf = oa_nodes_multi['oa']
    oa_nodes = list(oa_nodes_gdf.index)
    nx_graphs = {}
    for rel in ['contig', '15_min_walk', '15_min_multi']:
        G = c2g.gdf_to_nx(nodes=oa_nodes_gdf, edges=oa_edges_multi[('oa', rel, 'oa')])
        G = nx.relabel_nodes(G, {nid: attrs['_original_index'] for nid, attrs in G.nodes(data=True)})
        nx_graphs[rel] = G

    def get_embeddings(gdf, model_type):
        if model_type == 'baseline':
            feat_cols = [c for c in gdf.columns if c not in ['geometry', 'cluster', 'cluster_label_rank'] and gdf[c].dtype in ['float64', 'float32', 'int64']]
            Z = PCA(n_components=len(EMBEDDING_COLS), random_state=config['seeds']['global']).fit_transform(gdf[feat_cols].values)
        else:
            Z = gdf[EMBEDDING_COLS].values
        return normalize(Z, norm='l2')

    results = {}
    for model_name, gdf in model_gdfs.items():
        Z = get_embeddings(gdf, MODELS[model_name]['type'])
        rows = []
        for k in K_RANGE:
            labels = KMeans(n_clusters=k, random_state=config['seeds']['clustering'], n_init=10).fit_predict(Z)
            mods = {'contig': np.nan, '15_min_walk': np.nan, '15_min_multi': np.nan}
            if len(oa_nodes) == len(labels):
                comms = [set() for _ in range(k)]
                for i, c in enumerate(labels):
                    comms[c].add(oa_nodes[i])
                for rel, G in nx_graphs.items():
                    mods[rel] = nx.community.modularity(G, comms)
            rows.append({'k': k, 'silhouette': silhouette_score(Z, labels),
                         'calinski_harabasz': calinski_harabasz_score(Z, labels),
                         'davies_bouldin': davies_bouldin_score(Z, labels),
                         'modularity_contig': mods['contig'], 'modularity_walk': mods['15_min_walk'],
                         'modularity_multi': mods['15_min_multi']})
        k_df = pd.DataFrame(rows)
        best_k = int(k_df.loc[k_df['silhouette'].idxmax(), 'k'])
        results[model_name] = {'k_search': k_df, 'k': best_k}
        print(f"  {model_name}: best_k={best_k}")
    return results


def load_results(force=False):
    if not force and os.path.exists(CACHE):
        with open(CACHE, 'rb') as f:
            return pickle.load(f)
    print("Computing clustering (this takes a few minutes)...")
    results = compute_clustering()
    with open(CACHE, 'wb') as f:
        pickle.dump(results, f)
    return results


MODEL_COLORS = {
    "Baseline (PCA)": "#7f7f7f", "Model 1 (GAT-GAE)": "#1f77b4",
    "Model 2 (HAN-GAE: Walk)": "#ff7f0e", "Model 3 (HAN-GAE: Multi)": "#2ca02c",
    "Model 4 (HAN-GAE: All)": "#9467bd", "Model 1S (GAT-GAE: Struct)": "#1f77b4",
    "Model 2S (HAN-GAE: Walk-S)": "#ff7f0e", "Model 3S (HAN-GAE: Multi-S)": "#2ca02c",
    "Model 4S (HAN-GAE: All-S)": "#9467bd",
}
MODEL_MARKERS = {
    "Baseline (PCA)": "o", "Model 1 (GAT-GAE)": "s", "Model 2 (HAN-GAE: Walk)": "^",
    "Model 3 (HAN-GAE: Multi)": "D", "Model 4 (HAN-GAE: All)": "v",
    "Model 1S (GAT-GAE: Struct)": "s", "Model 2S (HAN-GAE: Walk-S)": "^",
    "Model 3S (HAN-GAE: Multi-S)": "D", "Model 4S (HAN-GAE: All-S)": "v",
}


def plot_v1(results, metric_col, ylabel, save_filename, struct_solid=False, struct_alpha=0.3):
    """v1 submitted style. If struct_solid, the dashed lines become solid;
    struct_alpha controls structure-only line opacity (v1 used 0.3)."""
    struct_ls = "-" if struct_solid else "--"
    struct_models = [m for m in results if "S (" in m]
    other_models = [m for m in results if "S (" not in m]

    fig, ax = plt.subplots(figsize=(10, 6))
    for m in other_models:
        k_df = results[m]['k_search']
        if metric_col not in k_df.columns or k_df[metric_col].isna().all():
            continue
        ax.plot(k_df['k'], k_df[metric_col], marker=MODEL_MARKERS[m], markersize=4,
                color=MODEL_COLORS[m], label=m, linewidth=1, linestyle="-", alpha=0.9)
    for m in struct_models:
        k_df = results[m]['k_search']
        if metric_col not in k_df.columns or k_df[metric_col].isna().all():
            continue
        ax.plot(k_df['k'], k_df[metric_col], marker=MODEL_MARKERS[m], markersize=4,
                color=MODEL_COLORS[m], label=m, linewidth=1, linestyle=struct_ls, alpha=struct_alpha)

    ax.set_xlabel('Number of Clusters (k)', fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)

    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))
    proposed = ["Model 1 (GAT-GAE)", "Model 2 (HAN-GAE: Walk)", "Model 3 (HAN-GAE: Multi)", "Model 4 (HAN-GAE: All)"]
    struct = ["Model 1S (GAT-GAE: Struct)", "Model 2S (HAN-GAE: Walk-S)", "Model 3S (HAN-GAE: Multi-S)", "Model 4S (HAN-GAE: All-S)"]
    ordered_handles, ordered_labels = [], []
    col3 = [("Baseline (PCA)", label_to_handle.get("Baseline (PCA)", Line2D([], [], linestyle="none")))] + \
           [("", Line2D([], [], linestyle="none")) for _ in range(3)]
    for i in range(4):
        ordered_handles.append(label_to_handle.get(proposed[i], Line2D([], [], linestyle="none")))
        ordered_labels.append(proposed[i])
    for i in range(4):
        ordered_handles.append(label_to_handle.get(struct[i], Line2D([], [], linestyle="none")))
        ordered_labels.append(struct[i])
    for i in range(4):
        ordered_handles.append(col3[i][1])
        ordered_labels.append(col3[i][0])
    legend = ax.legend(handles=ordered_handles, labels=ordered_labels, loc="lower center",
                       bbox_to_anchor=(0.5, -0.41), ncol=3, fontsize=12, title_fontsize=14,
                       frameon=True, facecolor="white", framealpha=1, edgecolor="lightgray",
                       columnspacing=1.0, handletextpad=0.5)
    for text in legend.get_texts():
        if text.get_text() == '' or (text.get_text() not in labels and text.get_text() != 'Baseline (PCA)'):
            text.set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    plt.savefig(f"{FIGURES_DIR}/{save_filename}", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  saved {save_filename}")


def plot_v2(results, metric_col, ylabel, save_filename,
            struct_lw=2.2, struct_ms=6, struct_alpha=0.95):
    """The "bold" single-panel style used in the revised manuscript.

    Keeps all nine series in one panel (no faceting) but makes the structure-only
    family (1S--4S) easy to read: full opacity, thicker dashed lines, and larger
    open (white-filled) markers, so the individual point symbols are legible while
    staying distinct from the solid-line proposed models.
    """
    struct_models = [m for m in results if "S (" in m]
    other_models = [m for m in results if "S (" not in m]

    fig, ax = plt.subplots(figsize=(10, 6))
    for m in other_models:
        k_df = results[m]['k_search']
        if metric_col not in k_df.columns or k_df[metric_col].isna().all():
            continue
        ax.plot(k_df['k'], k_df[metric_col], marker=MODEL_MARKERS[m], markersize=4,
                color=MODEL_COLORS[m], label=m, linewidth=1, linestyle="-", alpha=0.9)
    for m in struct_models:
        k_df = results[m]['k_search']
        if metric_col not in k_df.columns or k_df[metric_col].isna().all():
            continue
        ax.plot(k_df['k'], k_df[metric_col], marker=MODEL_MARKERS[m], markersize=struct_ms,
                markerfacecolor="white", markeredgecolor=MODEL_COLORS[m], markeredgewidth=1.2,
                color=MODEL_COLORS[m], label=m, linewidth=struct_lw, linestyle=(0, (5, 2)),
                alpha=struct_alpha)

    ax.set_xlabel('Number of Clusters (k)', fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)

    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))
    proposed = ["Model 1 (GAT-GAE)", "Model 2 (HAN-GAE: Walk)", "Model 3 (HAN-GAE: Multi)", "Model 4 (HAN-GAE: All)"]
    struct = ["Model 1S (GAT-GAE: Struct)", "Model 2S (HAN-GAE: Walk-S)", "Model 3S (HAN-GAE: Multi-S)", "Model 4S (HAN-GAE: All-S)"]
    ordered_handles, ordered_labels = [], []
    col3 = [("Baseline (PCA)", label_to_handle.get("Baseline (PCA)", Line2D([], [], linestyle="none")))] + \
           [("", Line2D([], [], linestyle="none")) for _ in range(3)]
    for i in range(4):
        ordered_handles.append(label_to_handle.get(proposed[i], Line2D([], [], linestyle="none")))
        ordered_labels.append(proposed[i])
    for i in range(4):
        ordered_handles.append(label_to_handle.get(struct[i], Line2D([], [], linestyle="none")))
        ordered_labels.append(struct[i])
    for i in range(4):
        ordered_handles.append(col3[i][1])
        ordered_labels.append(col3[i][0])
    legend = ax.legend(handles=ordered_handles, labels=ordered_labels, loc="lower center",
                       bbox_to_anchor=(0.5, -0.41), ncol=3, fontsize=12, title_fontsize=14,
                       frameon=True, facecolor="white", framealpha=1, edgecolor="lightgray",
                       columnspacing=1.0, handletextpad=0.5)
    for text in legend.get_texts():
        if text.get_text() == '' or (text.get_text() not in labels and text.get_text() != 'Baseline (PCA)'):
            text.set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    plt.savefig(f"{FIGURES_DIR}/{save_filename}", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  saved {save_filename}")


PANELS = [
    ('silhouette', 'Silhouette Score', 'silhouette.jpg'),
    ('modularity_contig', 'Modularity', 'modularity_contig.jpg'),
    ('modularity_walk', 'Modularity', 'modularity_walk.jpg'),
    ('modularity_multi', 'Modularity', 'modularity_multi.jpg'),
    ('davies_bouldin', 'Davies-Bouldin Index', 'davies_bouldin.jpg'),
    ('calinski_harabasz', 'Calinski-Harabasz Index', 'calinski_harabasz.jpg'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--style', choices=['v1', 'v1_solid', 'v1_solid_opaque', 'v1_opaque', 'bold'],
                    default='bold')
    ap.add_argument('--force', action='store_true', help='recompute clustering cache')
    ap.add_argument('--suffix', default='', help='filename suffix for preview (e.g. _v1)')
    args = ap.parse_args()
    results = load_results(force=args.force)
    # bold: the style used in the revised manuscript -- structure-only series drawn
    # at full opacity as thicker dashed lines with larger open (white-filled) markers,
    # so their point symbols are legible while staying distinct from the proposed
    # models. All nine models remain in a single panel per metric (no faceting).
    if args.style == 'bold':
        for metric, ylabel, fname in PANELS:
            if args.suffix:
                base, ext = os.path.splitext(fname)
                fname = f"{base}{args.suffix}{ext}"
            plot_v2(results, metric, ylabel, fname, struct_lw=2.2, struct_ms=6, struct_alpha=0.95)
        return
    # v1_opaque: keep v1's dashed structure-only lines (so they stay distinguishable
    # from the solid proposed models) but raise their opacity from 0.3 to 0.85 so
    # they are actually visible. This is the style used in the manuscript.
    struct_solid = args.style in ('v1_solid', 'v1_solid_opaque')
    struct_alpha = {'v1': 0.3, 'v1_solid': 0.3, 'v1_solid_opaque': 0.8, 'v1_opaque': 0.85}[args.style]
    for metric, ylabel, fname in PANELS:
        if args.suffix:
            base, ext = os.path.splitext(fname)
            fname = f"{base}{args.suffix}{ext}"
        plot_v1(results, metric, ylabel, fname, struct_solid=struct_solid, struct_alpha=struct_alpha)


if __name__ == '__main__':
    main()
