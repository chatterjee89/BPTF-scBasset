# scBasset — BPTF KO vs WT Mammary Epithelial Cells

## What this does
Trains a CNN (scBasset, Yuan & Kelley 2022) on your scATAC-seq data to:
1. Learn which DNA sequence features (TF motifs) drive chromatin accessibility per cell type
2. Generate sequence-based cell embeddings (UMAP)
3. Identify which motifs change in BPTF KO vs WT via in silico mutagenesis (ISM)

## Folder structure
```
scBasset/
├── data/
│   ├── genome/mm10.fa          ← downloaded by setup script
│   ├── atac_for_scbasset.h5ad  ← output of 01_prepare_data.py
│   ├── peaks_filtered.bed
│   └── sequences.h5            ← output of 02_extract_sequences.py
├── models/
│   └── scbasset_model.h5       ← best checkpoint from training
├── logs/
│   ├── training_log.csv
│   └── training_history.json
├── outputs/
│   ├── embeddings/             ← UMAP plots, training curves
│   ├── motifs/                 ← ISM logos, KO vs WT difference plots
│   └── variant_effects/        ← (future use)
└── scripts/
    ├── 00_setup_environment.sh
    ├── 01_prepare_data.py
    ├── 02_extract_sequences.py
    ├── 03_train_scbasset.py
    ├── 04_embeddings.py
    └── 05_motif_analysis.py
```

## Steps

### Step 0: One-time setup (run once)
```bash
bash scripts/00_setup_environment.sh
```
This creates the `scbasset` conda environment, installs TensorFlow + scBasset,
and downloads the mm10 genome FASTA (~850 MB download, ~3 GB unpacked).

### Step 1: Prepare data
```bash
conda activate scbasset
python scripts/01_prepare_data.py
```
Loads your pySCENIC-exported ATAC matrix, filters low-coverage peaks,
saves `atac_for_scbasset.h5ad` and `peaks_filtered.bed`.

### Step 2: Extract sequences
```bash
python scripts/02_extract_sequences.py
```
Extracts a 1344 bp window centered on each peak from mm10.fa,
one-hot encodes them, saves to `sequences.h5`.

### Step 3: Train
```bash
python scripts/03_train_scbasset.py
```
Trains the CNN. On CPU: ~2-4 hours. On GPU: ~30-60 min.
Best model saved to `models/scbasset_model.h5`.

### Step 4: Cell embeddings & UMAP
```bash
python scripts/04_embeddings.py
```
Extracts cell embeddings from the trained model → UMAP.
Outputs: `outputs/embeddings/umap_scbasset_*.pdf`

### Step 5: Motif analysis (ISM)
```bash
python scripts/05_motif_analysis.py
```
Runs ISM on top differentially accessible peaks per cell type.
Outputs: ISM logo plots and KO vs WT difference tracks.

## Input data used
- Peak matrix: `pySCENIC/data/ATAC/peak_counts_binary.mtx`
- Barcodes: `pySCENIC/data/ATAC/barcodes.txt`
- Peaks: `pySCENIC/data/ATAC/peaks.txt`
- Metadata: `pySCENIC/data/combined_metadata.csv`
- Genome: `data/genome/mm10.fa` (downloaded by setup)
