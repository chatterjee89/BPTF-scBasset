#!/usr/bin/env bash
# scBasset environment setup
# Run once before any other scripts.
# Usage: bash 00_setup_environment.sh

set -e

SCBASSET_DIR="$HOME/Documents/Coding/BPTF_scDataset/scBasset"
DATA_DIR="$SCBASSET_DIR/data"

# ── 1. Create conda environment ───────────────────────────────────────────────
echo "=== Creating scbasset conda environment ==="
conda create -y -n scbasset python=3.8
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate scbasset

# ── 2. Install TensorFlow (Apple Silicon: use tensorflow-macos + metal) ──────
echo "=== Installing TensorFlow ==="
pip install "tensorflow==2.13.*"

# ── 3. Install scBasset from source ──────────────────────────────────────────
echo "=== Installing scBasset ==="
pip install git+https://github.com/calico/scBasset.git

# ── 4. Install remaining dependencies ────────────────────────────────────────
echo "=== Installing dependencies ==="
pip install \
    anndata \
    scanpy \
    pyfaidx \
    kipoiseq \
    scipy \
    pandas \
    numpy \
    matplotlib \
    seaborn \
    h5py \
    tqdm

# ── 5. Download mm10 genome FASTA ─────────────────────────────────────────────
# Source: UCSC mm10 (GRCm38) — same assembly used for your Signac/SnapATAC2 analysis
echo "=== Downloading mm10 genome FASTA ==="
mkdir -p "$DATA_DIR/genome"
cd "$DATA_DIR/genome"

if [ ! -f mm10.fa ]; then
    curl -O https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz
    echo "Decompressing (this takes a few minutes)..."
    gunzip mm10.fa.gz
    echo "mm10.fa ready."
else
    echo "mm10.fa already exists, skipping download."
fi

# Index the FASTA for fast sequence extraction
python -c "from pyfaidx import Fasta; Fasta('mm10.fa'); print('FASTA indexed.')"

echo ""
echo "=== Setup complete ==="
echo "Activate environment with: conda activate scbasset"
echo "Then run scripts in order: 01 → 02 → 03 → 04 → 05"
