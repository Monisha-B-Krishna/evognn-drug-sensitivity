# Evolutionary GNN Framework for Drug Sensitivity Classification
### Monisha B K (2548417) & Kiruba Samuel I V (2548432)
### CHRIST (Deemed to be University), BYC — Supervisor: Dr. Kannan M

---

## Project Structure

```
evolutionary_gnn_drug_sensitivity/
│
├── data/
│   ├── raw/
│   │   ├── expression/              ← RNA-Seq gene expression
│   │   ├── drug_response/           ← GDSC2 IC50 values
│   │   ├── drug_annotations/        ← SMILES + fingerprints
│   │   ├── cell_line_metadata/      ← Cancer type, tissue info
│   │   └── ppi_network/             ← STRING PPI graph edges
│   └── processed/
│       ├── graphs/                  ← PyG graph objects
│       ├── features/                ← Processed feature matrices
│       └── labels/                  ← Sensitive/Resistant labels
│
├── src/
│   ├── preprocessing/
│   │   ├── organize_datasets.py     ← Step 1: Verify all files
│   │   ├── download_smiles.py       ← Step 2: Fetch drug SMILES
│   │   ├── generate_fingerprints.py ← Step 3: Morgan fingerprints
│   │   └── build_graphs.py          ← Step 4: Build PyG graphs
│   ├── models/                      ← GAT model definition
│   ├── training/                    ← GA + training loop
│   ├── evaluation/                  ← Metrics, baselines
│   └── utils/                       ← Helper functions
│
├── notebooks/                       ← Jupyter EDA notebooks
├── outputs/
│   ├── models/                      ← Saved model checkpoints
│   ├── results/                     ← Evaluation metrics
│   └── plots/                       ← Figures for paper
├── logs/                            ← Training logs
└── README.md
```

---

## Dataset Sources

| Dataset | Source | Purpose |
|---|---|---|
| rnaseq_merged_20260323.zip | cellmodelpassports.sanger.ac.uk | Gene expression (TPM) |
| GDSC2_dataset.csv | cellmodelpassports.sanger.ac.uk | IC50 drug response labels |
| screened_compounds.csv | cellmodelpassports.sanger.ac.uk | Drug PubChem IDs |
| model_list.csv | cellmodelpassports.sanger.ac.uk | Cell line metadata |
| 9606.protein.links.v12.0.txt.gz | string-db.org | PPI graph edges |
| 9606.protein.info.v12.0.txt.gz | string-db.org | Protein → gene name map |
| drug_smiles.csv | Auto-fetched via PubChem API | Drug SMILES strings |
| drug_fingerprints.csv | Auto-generated via RDKit | 2048-bit Morgan vectors |

---

## Setup Instructions

### 1. Install Dependencies
```bash
pip install torch torch-geometric pandas numpy scikit-learn
pip install rdkit requests tqdm matplotlib seaborn
pip install deap  # for Genetic Algorithm
```

### 2. Place Downloaded Files
```
data/raw/expression/         → rnaseq_merged_20260323.zip
data/raw/drug_response/      → GDSC2_dataset.csv
data/raw/drug_annotations/   → screened_compounds.csv
data/raw/cell_line_metadata/ → model_list.csv
data/raw/ppi_network/        → 9606.protein.links.v12.0.txt.gz
                             → 9606.protein.info.v12.0.txt.gz
```

### 3. Verify All Files
```bash
python src/preprocessing/organize_datasets.py
```

### 4. Download Drug SMILES (Automatic)
```bash
python src/preprocessing/download_smiles.py
```

### 5. Generate Drug Fingerprints
```bash
python src/preprocessing/generate_fingerprints.py
```

### 6. Build Graph Objects (Coming in next phase)
```bash
python src/preprocessing/build_graphs.py
```

---

## Pipeline Overview

```
Raw Data
   │
   ├─ RNA-Seq Expression ──────────────────────────┐
   ├─ GDSC2 IC50 → Sensitive/Resistant labels      │
   ├─ STRING PPI → Graph edges                     ▼
   ├─ Drug SMILES → Morgan Fingerprints     [Graph Construction]
   └─ Cell Line Metadata                           │
                                                   ▼
                                    [GA Feature Selection]
                                    Chromosome = gene subset
                                           │
                                           ▼
                                    [GAT Model Training]
                                    Graph Attention Network
                                    + Drug fingerprint concat
                                           │
                                           ▼
                                    [Classification Output]
                                    Sensitive vs Resistant
```
