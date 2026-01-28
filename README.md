# Case Study for City2Graph
Liverpool case study for City2Graph.

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
│   ├── .gitkeep
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
│   └── 05_visualization.ipynb
├── notebooks_samples
│   ├── data
│   ├── morphology.ipynb
│   ├── morphology_combined.png
│   ├── morphology_graph.png
│   ├── morphology_steps.png
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
├── tests
└── uv.lock
```

## Data (Zenodo)
The full data directory is hosted on Zenodo:

Sato, Y. (2026). Case Study Data for City2Graph: Clustering Urban Functions in Liverpool [Data set]. Zenodo. https://doi.org/10.5281/zenodo.18396286

Download the Zenodo archive and unzip it to the repository root so the `data/` directory matches the expected structure.


## Data sources and copyright

| Source | Data used | License / attribution | Source URL(s) |
| --- | --- | --- | --- |
| Office for National Statistics (ONS) | Output Areas (Dec 2021) EW BGC V2 boundaries; Output Areas (Dec 2021) population-weighted centroids V3 |   Open Government Licence v3.0; Contains OS data © Crown copyright and database right 2023 (boundaries). © Crown copyright and database right 2024 (centroids). See https://www.ons.gov.uk/methodology/geography/licences. | https://geoportal.statistics.gov.uk/datasets/6beafcfd9b9c4c9993a06b6b199d7e6d_0; https://geoportal.statistics.gov.uk/datasets/ons::output-areas-december-2021-ew-population-weighted-centroids-v3 |
| Overture Maps Foundation | Places (POIs), Base (land_use), Transportation (segment + connector), release 2025-12-17.0 | © OpenStreetMap contributors, Overture Maps Foundation. Accessed on Janurary 28th, 2026. See https://docs.overturemaps.org/attribution/. | https://overturemaps.org|
| UK Department for Transport (DfT) | Bus Open Data (GTFS timetables), North West feed (accessed Dec 10, 2025) | Open Government Licence v3.0; © Crown copyright. See https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/. | https://findtransportdata.dft.gov.uk/dataset/bus-open-data---download-all-timetable-data--18335fb19c4 |
| Metropolitan Transportation Authority (MTA) | GTFS schedules for NYC Subway (used in notebook samples) | Use is subject to MTA data feed terms and conditions. See https://www.mta.info/developers/terms-and-conditions | https://www.mta.info/developers|
| NY Open Data | MTA Subway Origin–Destination Ridership Estimate: Beginning 2025 (used in notebook samples) | Attribution in dataset metadata: “Metropolitan Transportation Authority”, with attribution link https://www.mta.info/open-data. | https://data.ny.gov/Transportation/MTA-Subway-Origin-Destination-Ridership-Estimate-B/y2qv-fytt |

## Models and baselines
- `GATGAE`: 2-layer GAT encoder with DistMult structure decoder for the homogeneous contiguity graph.
- `HANGAE`: 2-layer HAN encoder with semantic attention across metapaths, DistMult per relation.
- `run_kmeans`: K-Means clustering for embeddings and baseline feature clustering.

## Quickstart (notebooks)
0. Prepare for the data in [data/](data)
1. Run [notebooks/01_data_processing.ipynb](notebooks/01_data_processing.ipynb)
2. Run [notebooks/02_graph_construction.ipynb](notebooks/02_graph_construction.ipynb)
3. Run [notebooks/03_model_training.ipynb](notebooks/03_model_training.ipynb)
4. Run [notebooks/04_evaluation.ipynb](notebooks/04_evaluation.ipynb)
5. Run [notebooks/05_visualization.ipynb](notebooks/05_visualization.ipynb)

## Outputs
Results (embeddings, clusters, tables, and figures) are written under data/outputs/.
