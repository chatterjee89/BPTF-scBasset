"""
02_extract_sequences.py
Extract 1344bp sequences centered on each peak → save as h5 for scBasset training.
Requires mm10.fa (download via 00_setup_environment.sh).
"""

import os
import h5py
import numpy as np
import anndata
from pyfaidx import Fasta
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.expanduser(
    "~/Documents/Coding/BPTF_scDataset/scBasset/data"
)
FASTA_PATH = os.path.join(DATA_DIR, "genome", "mm10.fa")
ADATA_PATH = os.path.join(DATA_DIR, "atac_for_scbasset.h5ad")
OUT_H5    = os.path.join(DATA_DIR, "sequences.h5")

SEQ_LEN = 1344   # scBasset fixed input length

# ── One-hot encoding ──────────────────────────────────────────────────────────
NUCLEOTIDE_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}

def one_hot_encode(seq):
    """Convert DNA string to (SEQ_LEN, 4) one-hot array."""
    seq = seq.upper()
    ohe = np.zeros((len(seq), 4), dtype=np.float32)
    for i, base in enumerate(seq):
        if base in NUCLEOTIDE_MAP:
            ohe[i, NUCLEOTIDE_MAP[base]] = 1.0
        # N or ambiguous → all zeros (already set)
    return ohe

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading AnnData...")
adata = anndata.read_h5ad(ADATA_PATH)
print(f"  {adata.n_vars} peaks x {adata.n_obs} cells")

print("Loading mm10 FASTA...")
fasta = Fasta(FASTA_PATH, as_raw=True, sequence_always_upper=True)
print("  FASTA loaded.")

# ── Extract sequences ─────────────────────────────────────────────────────────
n_peaks = adata.n_vars
sequences = np.zeros((n_peaks, SEQ_LEN, 4), dtype=np.float32)

skipped = 0
for i, (peak_id, row) in enumerate(tqdm(adata.var.iterrows(), total=n_peaks, desc="Extracting sequences")):
    chrom = row["chrom"]
    center = (row["start"] + row["end"]) // 2
    seq_start = center - SEQ_LEN // 2
    seq_end   = seq_start + SEQ_LEN

    # Skip peaks too close to chromosome edge
    if seq_start < 0:
        skipped += 1
        continue

    try:
        chrom_len = len(fasta[chrom])
        if seq_end > chrom_len:
            skipped += 1
            continue
        seq = fasta[chrom][seq_start:seq_end]
        sequences[i] = one_hot_encode(seq)
    except KeyError:
        skipped += 1   # chromosome not in FASTA (e.g. random contigs)

print(f"\nSkipped {skipped} peaks (out of bounds or missing chrom)")
print(f"Sequences shape: {sequences.shape}")

# ── Save to h5 ────────────────────────────────────────────────────────────────
print(f"\nSaving to {OUT_H5}...")
with h5py.File(OUT_H5, "w") as f:
    f.create_dataset("sequences", data=sequences, compression="gzip", chunks=(1000, SEQ_LEN, 4))
    # Store peak IDs for reference
    f.create_dataset(
        "peak_ids",
        data=np.array(adata.var_names.tolist(), dtype=h5py.special_dtype(vlen=str))
    )

print(f"Saved: {OUT_H5}")
print("\nDone. Next: run 03_train_scbasset.py")
