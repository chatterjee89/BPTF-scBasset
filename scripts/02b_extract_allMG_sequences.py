"""
02b_extract_allMG_sequences.py
Extract 1344bp sequences for all peaks in the full mammary gland dataset.
Mirrors 02_extract_sequences.py but uses allMG data.
"""

import os
import h5py
import numpy as np
import anndata
from pyfaidx import Fasta
from tqdm import tqdm

DATA_DIR  = os.path.expanduser("~/Documents/Coding/BPTF_scDataset/scBasset/data/allMG")
FASTA     = os.path.expanduser(
    "~/Documents/Coding/BPTF_scDataset/scBasset/data/genome/mm10.fa"
)
ADATA_PATH = os.path.join(DATA_DIR, "allMG_for_scbasset.h5ad")
OUT_H5     = os.path.join(DATA_DIR, "allMG_sequences.h5")
SEQ_LEN    = 1344
NUCLEOTIDE_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}

def one_hot_encode(seq):
    ohe = np.zeros((len(seq), 4), dtype=np.float32)
    for i, base in enumerate(seq.upper()):
        if base in NUCLEOTIDE_MAP:
            ohe[i, NUCLEOTIDE_MAP[base]] = 1.0
    return ohe

print("Loading AnnData...")
adata = anndata.read_h5ad(ADATA_PATH)
print(f"  {adata.n_vars} peaks x {adata.n_obs} cells")

print("Loading mm10 FASTA...")
fasta = Fasta(FASTA, as_raw=True, sequence_always_upper=True)

n_peaks   = adata.n_vars
sequences = np.zeros((n_peaks, SEQ_LEN, 4), dtype=np.float32)
skipped   = 0

for i, (peak_id, row) in enumerate(tqdm(adata.var.iterrows(), total=n_peaks,
                                         desc="Extracting sequences")):
    chrom  = row["chrom"]
    center = (row["start"] + row["end"]) // 2
    s, e   = center - SEQ_LEN // 2, center - SEQ_LEN // 2 + SEQ_LEN
    if s < 0:
        skipped += 1; continue
    try:
        if e > len(fasta[chrom]):
            skipped += 1; continue
        sequences[i] = one_hot_encode(fasta[chrom][s:e])
    except KeyError:
        skipped += 1

print(f"\nSkipped: {skipped} peaks")
print(f"Sequences shape: {sequences.shape}")

print(f"Saving to {OUT_H5}...")
with h5py.File(OUT_H5, "w") as f:
    f.create_dataset("sequences", data=sequences,
                     compression="gzip", chunks=(1000, SEQ_LEN, 4))
    f.create_dataset("peak_ids",
                     data=np.array(adata.var_names.tolist(),
                                   dtype=h5py.special_dtype(vlen=str)))

print(f"Saved: {OUT_H5}")
print("Done. Next: run 03b_train_allMG.py")
