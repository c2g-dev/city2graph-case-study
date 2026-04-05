# Case Study for City2Graph
Liverpool case study for [City2Graph](https://github.com/c2g-dev/city2graph).

<p align="center">
  <img width="100%" alt="Case Study of City2Graph" src="https://github.com/user-attachments/assets/a824428c-11ab-4e8c-88a7-4474f704f866" />
</p>

## Repository structure


```
city2graph-case-study
├── .gitignore
├── .python-version
├── .vscode
│   └── settings.json
├── README.md
├── configs
│   └── experiment_config.yaml
├── data
│   ├── outputs
│   │   ├── checkpoints
│   │   ├── clusters
│   │   ├── embeddings
│   │   ├── figures
│   │   └── tables
│   ├── processed
│   │   ├── features
│   │   ├── graphs
│   │   └── isochrones
│   └── raw
│       ├── gtfs
│       ├── output_area
│       └── overture
├── notebooks
│   ├── 01_data_processing.ipynb
│   ├── 02_graph_construction.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_evaluation.ipynb
│   ├── 05_visualization.ipynb
│   └── appendix_evaluation_hdbscan.ipynb
├── notebooks_samples
│   ├── data
│   ├── morphology.ipynb
│   ├── morphology_combined.jpg
│   ├── morphology_graph.jpg
│   ├── morphology_steps.jpg
│   └── transportation_mobility.ipynb
├── pyproject.toml
├── src
│   ├── baselines
│   │   ├── __init__.py
│   │   └── kmeans.py
│   └── models
│       ├── __init__.py
│       ├── gat_gae.py
│       ├── han_gae.py
│       └── utils.py
└── uv.lock
```

## Data (Zenodo)
The full data directory is hosted on Zenodo:

Sato, Y. (2026). Case Study Data for City2Graph: Clustering Urban Functions in Liverpool [Data set]. Zenodo. https://doi.org/10.5281/zenodo.18396286

Download the Zenodo archive and unzip it to the repository root so the `data/` directory matches the expected structure.

## Models and baselines
- `GATGAE`: 2-layer GAT encoder with DistMult structure decoder for the homogeneous contiguity graph.
- `HANGAE`: 2-layer HAN encoder with semantic attention across metapaths, DistMult per relation.
- `run_kmeans`: K-Means clustering for embeddings and baseline feature clustering.

## Quickstart (notebooks)

0. Prepare for the data in [data/](data)
1. Run [notebooks/01_data_processing.ipynb](notebooks/01_data_processing.ipynb) (Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1O2mHzH3JPbteL0mW63vetmFCcO45gjA4))

<p align="center">
  <img width="366.1" height="520" alt="fig8-1_land_use" src="https://github.com/user-attachments/assets/fe60a629-f022-4a42-a5d1-63cc38b4c406" />
  <img width="400" height="476" alt="fig8-2_poi" src="https://github.com/user-attachments/assets/ea7811aa-3c16-4aee-96f9-49f679fe329e" />
</p>


2. Run [notebooks/02_graph_construction.ipynb](notebooks/02_graph_construction.ipynb) / [noteboosk/05_visualization.ipynb](https://github.com/c2g-dev/city2graph-case-study/blob/main/notebooks/05_visualization.ipynb) (Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RStJI9xq6iDM7zz7_MQ18BgUmXHex7gB) / [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1t1fNdsKNFMH1BdmEKt7HUwN6pCQNgr8w))

<p align="center">
  <img width="100%" alt="fig9_liverpool_contig" src="https://github.com/user-attachments/assets/60962e17-fa11-405f-9cbe-536411bd43e3" />
</p>
<p align="center">
  <img width="100%" alt="fig10_liverpool_metapaths" src="https://github.com/user-attachments/assets/0107eb56-5145-4b29-8192-3ac613976060" />
</p>


3. Run [notebooks/03_model_training.ipynb](notebooks/03_model_training.ipynb) (Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ogAv8XrzzlTB1L49wfgd0TmSVquJKLL6))

<table align="center">
  <tr>
    <td>
      <img width="100%" alt="model-1" src="https://github.com/user-attachments/assets/779423ae-a68f-40df-8264-a5124a3b28c5" />
    </td>
    <td>
      <img width="100%" alt="model-2" src="https://github.com/user-attachments/assets/5a603cf9-8b6a-49d1-adf8-2f4f96d27874" />
    </td>
    <td>
      <img width="100%" alt="model-3" src="https://github.com/user-attachments/assets/31102c1d-704e-4339-8845-b62c11a61575" />
    </td>
    <td>
      <img width="100%" alt="model-4" src="https://github.com/user-attachments/assets/ac1911ec-3964-41f9-ba2b-ccc9e385bf0c" />
    </td>
  </tr>
  <tr>
    <td>
      <img width="100%" alt="model-1s" src="https://github.com/user-attachments/assets/243b56c8-ec43-4b2e-9e22-7ba4423f6488" />      
    </td>
    <td>
      <img width="100%" alt="model-2s" src="https://github.com/user-attachments/assets/ac7ce3f6-9c06-4b75-bb0b-5a3f7e4f6f6f" />
    </td>
    <td>
      <img width="100%" alt="model-3s" src="https://github.com/user-attachments/assets/118de576-1a26-484d-a1a6-74a35959299b" />
    </td>
    <td>
      <img width="100%" alt="model-4s" src="https://github.com/user-attachments/assets/a6245c48-833d-49a8-a608-b254e2023157" />
    </td>
  </tr>
</table>

4. Run [notebooks/04_evaluation.ipynb](notebooks/04_evaluation.ipynb) (Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1EMissX0vP7_THWnmn9tW-qakIpdJ3T7k))

<table align="center">
  <tr>
    <img width="100%" alt="clusters" src="https://github.com/user-attachments/assets/5d47a7db-311f-46dc-8512-2d6645fd8b1a" />
  </tr>
  <tr>
    <img width="100%" alt="similarity" src="https://github.com/user-attachments/assets/4b6d86b6-4514-4d9d-a430-8f313209607a" />
  </tr>
</table>

<p align="center">
  <img width="100%" alt="isochrones" src="https://github.com/user-attachments/assets/6658788a-ab6a-4e59-9033-134d0322edc6" />
</p>

## Outputs
Results (embeddings, clusters, tables, and figures) are written under data/outputs/.

## Reproducibility note
This case study uses uv for dependency management and environment reproducibility.

- Dependency specification: `pyproject.toml`
- Resolved, reproducible lockfile: `uv.lock`
- Python version pin: `.python-version` (3.12.8)

To reproduce the exact environment from this repository:

```bash
uv sync
```

To verify installed package versions in the uv environment:

```bash
uv run python - <<'PY'
from importlib.metadata import version

packages = [
  "city2graph",
  "contextily",
  "geopandas",
  "hdbscan",
  "ipykernel",
  "jupyter",
  "mapclassify",
  "matplotlib",
  "matplotlib-scalebar",
  "networkx",
  "numpy",
  "pandas",
  "PyYAML",
  "scikit-learn",
  "seaborn",
  "splot",
  "torch",
  "torch-geometric",
  "torchaudio",
  "torchvision",
]

for pkg in packages:
  print(f"{pkg}=={version(pkg)}")
PY
```

This case study was run on a CPU of Apple M2 (ARM) with 16 GB RAM, and CUDA was not used.

## Data sources and copyright

| Source | Data used | License / attribution | Source URL(s) |
| --- | --- | --- | --- |
| Office for National Statistics (ONS) | Output Areas (Dec 2021) EW BGC V2 boundaries; Output Areas (Dec 2021) population-weighted centroids V3 |   Open Government Licence v3.0; Contains OS data © Crown copyright and database right 2023 (boundaries). © Crown copyright and database right 2024 (centroids). See https://www.ons.gov.uk/methodology/geography/licences. | https://geoportal.statistics.gov.uk/datasets/6beafcfd9b9c4c9993a06b6b199d7e6d_0; https://geoportal.statistics.gov.uk/datasets/ons::output-areas-december-2021-ew-population-weighted-centroids-v3 |
| Overture Maps Foundation | Places (POIs), Base (land_use), Transportation (segment + connector), release 2025-12-17.0 | © OpenStreetMap contributors, Overture Maps Foundation. Accessed on Janurary 28th, 2026. See https://docs.overturemaps.org/attribution/. | https://overturemaps.org|
| UK Department for Transport (DfT) | Bus Open Data (GTFS timetables), North West feed (accessed Dec 10, 2025) | Open Government Licence v3.0; © Crown copyright. See https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/. | https://findtransportdata.dft.gov.uk/dataset/bus-open-data---download-all-timetable-data--18335fb19c4 |
| Metropolitan Transportation Authority (MTA) | GTFS schedules for NYC Subway (used in notebook samples) | Use is subject to MTA data feed terms and conditions. See https://www.mta.info/developers/terms-and-conditions | https://www.mta.info/developers|
| NY Open Data | MTA Subway Origin–Destination Ridership Estimate: Beginning 2025 (used in notebook samples) | Attribution in dataset metadata: “Metropolitan Transportation Authority”, with attribution link https://www.mta.info/open-data. | https://data.ny.gov/Transportation/MTA-Subway-Origin-Destination-Ridership-Estimate-B/y2qv-fytt |
