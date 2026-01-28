# Case Study for City2Graph
Liverpool case study for City2Graph.

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


2. Run [notebooks/02_graph_construction.ipynb](notebooks/02_graph_construction.ipynb) (Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RStJI9xq6iDM7zz7_MQ18BgUmXHex7gB))

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
      <img width="100%" alt="image" src="https://github.com/user-attachments/assets/f1ce994b-d871-4bc9-887e-7d721ec0537d" />
    </td>
    <td>
      <img width="100%" alt="image" src="https://github.com/user-attachments/assets/c302cf99-2d32-4ffa-9a3e-a77c246fd9c2" />
    </td>
  </tr>
  <tr>
    <td>
      <img width="100%" alt="image" src="https://github.com/user-attachments/assets/74e5ef1b-3db8-4d85-91fe-213ed63513fd" />
    </td>
    <td>
      <img width="100%" alt="image" src="https://github.com/user-attachments/assets/19997e97-9757-44d0-a8c7-3a8e7e4b3557" />
    </td>
  </tr>
</table>


4. Run [notebooks/04_evaluation.ipynb](notebooks/04_evaluation.ipynb) (Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1EMissX0vP7_THWnmn9tW-qakIpdJ3T7k))

<p align="center">
  <img width="100%" alt="fig13-1_cluster_maps" src="https://github.com/user-attachments/assets/297097b8-aba8-480e-8190-e146923df9d7" />
</p>

<p align="center">
  <img width="100%" alt="fig13-2_cluster_maps_similarity" src="https://github.com/user-attachments/assets/93f9053a-bc96-4683-b826-daaf6afd2faa" />
</p>


<p align="center">
  <img width="100%" alt="fig14_isochrones" src="https://github.com/user-attachments/assets/b3c77b90-b177-498a-af35-7206d38e1329" />
</p>


## Outputs
Results (embeddings, clusters, tables, and figures) are written under data/outputs/.


## Data sources and copyright

| Source | Data used | License / attribution | Source URL(s) |
| --- | --- | --- | --- |
| Office for National Statistics (ONS) | Output Areas (Dec 2021) EW BGC V2 boundaries; Output Areas (Dec 2021) population-weighted centroids V3 |   Open Government Licence v3.0; Contains OS data © Crown copyright and database right 2023 (boundaries). © Crown copyright and database right 2024 (centroids). See https://www.ons.gov.uk/methodology/geography/licences. | https://geoportal.statistics.gov.uk/datasets/6beafcfd9b9c4c9993a06b6b199d7e6d_0; https://geoportal.statistics.gov.uk/datasets/ons::output-areas-december-2021-ew-population-weighted-centroids-v3 |
| Overture Maps Foundation | Places (POIs), Base (land_use), Transportation (segment + connector), release 2025-12-17.0 | © OpenStreetMap contributors, Overture Maps Foundation. Accessed on Janurary 28th, 2026. See https://docs.overturemaps.org/attribution/. | https://overturemaps.org|
| UK Department for Transport (DfT) | Bus Open Data (GTFS timetables), North West feed (accessed Dec 10, 2025) | Open Government Licence v3.0; © Crown copyright. See https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/. | https://findtransportdata.dft.gov.uk/dataset/bus-open-data---download-all-timetable-data--18335fb19c4 |
| Metropolitan Transportation Authority (MTA) | GTFS schedules for NYC Subway (used in notebook samples) | Use is subject to MTA data feed terms and conditions. See https://www.mta.info/developers/terms-and-conditions | https://www.mta.info/developers|
| NY Open Data | MTA Subway Origin–Destination Ridership Estimate: Beginning 2025 (used in notebook samples) | Attribution in dataset metadata: “Metropolitan Transportation Authority”, with attribution link https://www.mta.info/open-data. | https://data.ny.gov/Transportation/MTA-Subway-Origin-Destination-Ridership-Estimate-B/y2qv-fytt |
