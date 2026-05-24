"""
01_prepare_data.py
Load pySCENIC-exported ATAC data → build AnnData → filter peaks → save for scBasset.
"""

import os
import scipy.io
import scipy.sparse
import pandas as pd
import numpy as np
import anndata

# ── Paths ─────────────────────────────────────────────────────────────────────
PYSCENIC_DATA = os.path.expanduser(
    "~/Documents/Coding/BPTF_scDataset/pySCENIC/data/ATAC"
)
METADATA_PATH = os.path.expanduser(
    "~/Documents/Coding/BPTF_scDataset/pySCENIC/data/metadata/combined_metadata.csv"
)
OUT_DIR = os.path.expanduser(
    "~/Documents/Coding/BPTF_scDataset/scBasset/data"
)
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load peak x cell matrix ───────────────────────────────────────────────────
print("Loading peak x cell matrix...")
# Use binary matrix (presence/absence of accessibility) as scBasset expects
mat = scipy.io.mmread(os.path.join(PYSCENIC_DATA, "peak_counts_binary.mtx"))
mat = scipy.sparse.csc_matrix(mat)   # peaks x cells

barcodes = pd.read_csv(
    os.path.join(PYSCENIC_DATA, "barcodes.txt"), header=None
)[0].tolist()

peaks = pd.read_csv(
    os.path.join(PYSCENIC_DATA, "peaks.txt"), header=None
)[0].tolist()

print(f"  Matrix shape: {mat.shape[0]} peaks x {mat.shape[1]} cells")

# ── Load metadata ─────────────────────────────────────────────────────────────
print("Loading metadata...")
meta = pd.read_csv(METADATA_PATH, index_col=0)
print(f"  Metadata: {meta.shape[0]} rows, columns: {meta.columns.tolist()}")

# ── Build AnnData (cells x peaks — scBasset convention) ──────────────────────
# Transpose so rows=cells, columns=peaks
adata = anndata.AnnData(
    X=mat.T.tocsr(),           # cells x peaks
    obs=pd.DataFrame(index=barcodes),
    var=pd.DataFrame(index=peaks)
)

# ── Attach metadata ───────────────────────────────────────────────────────────
# Match barcodes between metadata and adata
shared = adata.obs_names.intersection(meta.index)
print(f"  Shared barcodes: {len(shared)} / {adata.n_obs}")

adata = adata[shared].copy()
adata.obs = adata.obs.join(meta, how="left")

# Recode condition labels for display
if "orig.ident" in adata.obs.columns:
    adata.obs["condition"] = adata.obs["orig.ident"].map(
        {"WT2": "WT", "KO3": "KO"}
    )
elif "sample" in adata.obs.columns:
    adata.obs["condition"] = adata.obs["sample"].map(
        {"WT2": "WT", "KO3": "KO"}
    )

print("  Cell type counts:")
ct_col = [c for c in adata.obs.columns if "cell" in c.lower() or "type" in c.lower()]
if ct_col:
    print(adata.obs[ct_col[0]].value_counts().to_string())

# ── Parse peak coordinates into var DataFrame ─────────────────────────────────
# Peak format in your data: chr1-3119537-3120746
print("\nParsing peak coordinates...")
peak_df = adata.var_names.str.extract(r"^(chr\w+)[-:](\d+)[-:](\d+)$")
peak_df.columns = ["chrom", "start", "end"]
peak_df["start"] = peak_df["start"].astype(int)
peak_df["end"] = peak_df["end"].astype(int)
peak_df["peak_width"] = peak_df["end"] - peak_df["start"]
peak_df.index = adata.var_names

adata.var = peak_df
print(f"  Peaks parsed. Width stats:\n{peak_df['peak_width'].describe()}")

# ── Filter peaks: require accessibility in at least 1% of cells ──────────────
min_cells = max(10, int(0.01 * adata.n_obs))
peak_cell_counts = np.array((adata.X > 0).sum(axis=0)).flatten()
keep_peaks = peak_cell_counts >= min_cells

print(f"\nPeak filtering: keep peaks accessible in >= {min_cells} cells")
print(f"  Before: {adata.n_vars} peaks")
adata = adata[:, keep_peaks].copy()
print(f"  After:  {adata.n_vars} peaks")

# ── Filter peaks: remove non-standard chromosomes ────────────────────────────
standard_chroms = {f"chr{i}" for i in range(1, 20)} | {"chrX", "chrY"}
keep_chrom = adata.var["chrom"].isin(standard_chroms)
print(f"\nChromosome filter: removing {(~keep_chrom).sum()} peaks on non-standard chroms")
adata = adata[:, keep_chrom].copy()
print(f"  Final: {adata.n_vars} peaks x {adata.n_obs} cells")

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, "atac_for_scbasset.h5ad")
adata.write_h5ad(out_path)
print(f"\nSaved: {out_path}")

# Also save peaks.bed for sequence extraction (step 02)
bed_path = os.path.join(OUT_DIR, "peaks_filtered.bed")
adata.var[["chrom", "start", "end"]].to_csv(
    bed_path, sep="\t", header=False, index=True
)
print(f"Saved: {bed_path}")

print("\nDone. Next: run 02_extract_sequences.py")
