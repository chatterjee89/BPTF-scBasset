"""
01b_prepare_allMG_data.py
Prepare the full mammary gland dataset (all cell types, WT + KO) for scBasset retraining.
Mirrors 01_prepare_data.py but uses allMG export from 00b_export_allMG_data.R.
"""

import os
import scipy.io
import scipy.sparse
import pandas as pd
import numpy as np
import anndata

ATAC_DIR  = os.path.expanduser(
    "~/Documents/Coding/BPTF_scDataset/scBasset/data/allMG/ATAC"
)
META_PATH = os.path.expanduser(
    "~/Documents/Coding/BPTF_scDataset/scBasset/data/allMG/metadata/combined_metadata.csv"
)
OUT_DIR   = os.path.expanduser(
    "~/Documents/Coding/BPTF_scDataset/scBasset/data/allMG"
)

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading peak x cell matrix...")
mat      = scipy.io.mmread(os.path.join(ATAC_DIR, "peak_counts_binary.mtx"))
mat      = scipy.sparse.csc_matrix(mat)
barcodes = pd.read_csv(os.path.join(ATAC_DIR, "barcodes.txt"), header=None)[0].tolist()
peaks    = pd.read_csv(os.path.join(ATAC_DIR, "peaks.txt"),    header=None)[0].tolist()
print(f"  Matrix shape: {mat.shape[0]} peaks x {mat.shape[1]} cells")

meta = pd.read_csv(META_PATH, index_col=0)
print(f"  Metadata: {meta.shape[0]} cells")
print(f"  Cell types:\n{meta['cell_type'].value_counts().to_string()}")
print(f"  Conditions:\n{meta['condition'].value_counts().to_string()}")

# ── Build AnnData ─────────────────────────────────────────────────────────────
adata = anndata.AnnData(
    X   = mat.T.tocsr(),
    obs = pd.DataFrame(index=barcodes),
    var = pd.DataFrame(index=peaks)
)

shared = adata.obs_names.intersection(meta.index)
print(f"\nShared barcodes: {len(shared)} / {adata.n_obs}")
adata = adata[shared].copy()
adata.obs = adata.obs.join(meta, how="left")

# ── Parse peak coordinates ────────────────────────────────────────────────────
print("\nParsing peak coordinates...")
peak_df = adata.var_names.str.extract(r"^(chr\w+)[-:](\d+)[-:](\d+)$")
peak_df.columns = ["chrom", "start", "end"]
peak_df["start"] = peak_df["start"].astype(int)
peak_df["end"]   = peak_df["end"].astype(int)
peak_df.index    = adata.var_names
adata.var = peak_df

# ── Filter peaks ──────────────────────────────────────────────────────────────
min_cells = max(10, int(0.01 * adata.n_obs))
peak_cell_counts = np.array((adata.X > 0).sum(axis=0)).flatten()
keep = peak_cell_counts >= min_cells
print(f"\nPeak filter (>= {min_cells} cells): {adata.n_vars} → {keep.sum()} peaks")
adata = adata[:, keep].copy()

# Standard chromosomes only
standard_chroms = {f"chr{i}" for i in range(1, 20)} | {"chrX", "chrY"}
keep_chrom = adata.var["chrom"].isin(standard_chroms)
print(f"Chromosome filter: {(~keep_chrom).sum()} peaks removed")
adata = adata[:, keep_chrom].copy()
print(f"Final: {adata.n_vars} peaks x {adata.n_obs} cells")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\nCell type x condition breakdown:")
print(pd.crosstab(adata.obs["cell_type"], adata.obs["condition"]))

# ── Save ──────────────────────────────────────────────────────────────────────
out_h5   = os.path.join(OUT_DIR, "allMG_for_scbasset.h5ad")
out_bed  = os.path.join(OUT_DIR, "allMG_peaks_filtered.bed")
adata.write_h5ad(out_h5)
adata.var[["chrom", "start", "end"]].to_csv(out_bed, sep="\t", header=False, index=True)
print(f"\nSaved: {out_h5}")
print(f"Saved: {out_bed}")
print("\nDone. Next: run 02b_extract_allMG_sequences.py")
