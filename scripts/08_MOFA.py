"""
08_MOFA.py
Multi-Omics Factor Analysis (MOFA+) integrating RNA and scATAC-seq.
Decomposes both modalities into shared latent factors that explain
coordinated gene expression + chromatin accessibility variation.

Inputs:
  - RNA count matrix (pySCENIC/data/RNA/)
  - scBasset-predicted accessibility OR raw ATAC peak matrix
  - Cell metadata (cell type, condition)

Outputs:
  - Factor scores per cell (UMAP, heatmaps)
  - Top genes and peaks per factor
  - KO vs WT factor comparison per cell type

Run after 04_embeddings.py.
"""

import os
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse
import anndata
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# ── Paths ─────────────────────────────────────────────────────────────────────
RNA_DIR      = os.path.expanduser("~/Documents/Coding/BPTF_scDataset/pySCENIC/data/RNA")
ATAC_DIR     = os.path.expanduser("~/Documents/Coding/BPTF_scDataset/pySCENIC/data/ATAC")
SCBASSET_DIR = os.path.expanduser("~/Documents/Coding/BPTF_scDataset/scBasset")
OUT_DIR      = os.path.join(SCBASSET_DIR, "outputs", "MOFA")
os.makedirs(OUT_DIR, exist_ok=True)

META_PATH    = os.path.expanduser(
    "~/Documents/Coding/BPTF_scDataset/pySCENIC/data/metadata/combined_metadata.csv"
)
ATAC_H5AD    = os.path.join(SCBASSET_DIR, "data", "atac_for_scbasset_with_embeddings.h5ad")

CELLTYPE_COLORS  = {"LHS": "#4CAF50", "LASPs": "#FF9800", "BMyo": "#9C27B0", "BL": "#00BCD4"}
CONDITION_COLORS = {"WT": "#2196F3", "KO": "#F44336"}
N_FACTORS        = 15   # number of latent factors to learn

# ── Install mofapy2 if needed ─────────────────────────────────────────────────
try:
    import mofapy2
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mofapy2"])
    import mofapy2

from mofapy2.run.entry_point import entry_point

# ── Load RNA data ─────────────────────────────────────────────────────────────
print("Loading RNA data...")
rna_mat   = scipy.io.mmread(os.path.join(RNA_DIR, "counts.mtx"))
rna_mat   = scipy.sparse.csc_matrix(rna_mat)   # genes x cells
rna_genes = pd.read_csv(os.path.join(RNA_DIR, "genes.txt"), header=None)[0].tolist()
rna_bcs   = pd.read_csv(os.path.join(RNA_DIR, "barcodes.txt"), header=None)[0].tolist()

adata_rna = anndata.AnnData(
    X   = rna_mat.T.tocsr(),   # cells x genes
    obs = pd.DataFrame(index=rna_bcs),
    var = pd.DataFrame(index=rna_genes)
)
print(f"  RNA: {adata_rna.n_obs} cells x {adata_rna.n_vars} genes")

# ── Normalise RNA ─────────────────────────────────────────────────────────────
sc.pp.normalize_total(adata_rna, target_sum=1e4)
sc.pp.log1p(adata_rna)
sc.pp.highly_variable_genes(adata_rna, n_top_genes=5000)
adata_rna = adata_rna[:, adata_rna.var["highly_variable"]].copy()
print(f"  After HVG selection: {adata_rna.n_vars} genes")

# ── Load ATAC data ────────────────────────────────────────────────────────────
print("Loading ATAC data...")
adata_atac = anndata.read_h5ad(ATAC_H5AD)
print(f"  ATAC: {adata_atac.n_obs} cells x {adata_atac.n_vars} peaks")

# ── Find shared cells ─────────────────────────────────────────────────────────
shared = list(set(adata_rna.obs_names) & set(adata_atac.obs_names))
print(f"\nShared cells (RNA ∩ ATAC): {len(shared)}")

adata_rna  = adata_rna[shared].copy()
adata_atac = adata_atac[shared].copy()

# ── Attach metadata ───────────────────────────────────────────────────────────
meta = pd.read_csv(META_PATH, index_col=0)
adata_rna.obs  = adata_rna.obs.join(meta[["cell_type", "condition"]], how="left")
adata_atac.obs = adata_atac.obs.join(meta[["cell_type", "condition"]], how="left")

# ── Reduce ATAC dimensionality with LSI before passing to MOFA ───────────────
# MOFA can't handle 100k peaks directly — reduce to top 50 LSI components first
print("Running LSI on ATAC...")
sc.tl.pca(adata_atac, n_comps=50, use_highly_variable=False)
atac_lsi = adata_atac.obsm["X_pca"]   # cells x 50

# ── Build MOFA input: list of views ──────────────────────────────────────────
# MOFA expects: list of numpy arrays [n_cells x n_features] per view
rna_matrix  = adata_rna.X.toarray() if scipy.sparse.issparse(adata_rna.X) else adata_rna.X
atac_matrix = atac_lsi

# Scale each view to zero mean, unit variance (MOFA default)
def scale_view(X):
    mean = X.mean(axis=0)
    std  = X.std(axis=0) + 1e-8
    return (X - mean) / std

rna_scaled  = scale_view(rna_matrix).astype(np.float32)
atac_scaled = scale_view(atac_matrix).astype(np.float32)

print(f"  RNA view:  {rna_scaled.shape}")
print(f"  ATAC view: {atac_scaled.shape}")

# ── Run MOFA+ ─────────────────────────────────────────────────────────────────
print(f"\nRunning MOFA+ with {N_FACTORS} factors...")
ent = entry_point()

ent.set_data_options(scale_views=False)   # already scaled above
ent.set_data_matrix(
    data        = [[rna_scaled], [atac_scaled]],
    views_names = ["RNA", "ATAC"],
    groups_names= ["all"],
    samples_names = [shared],
    features_names = [adata_rna.var_names.tolist(), [f"LSI{i}" for i in range(50)]]
)

ent.set_model_options(factors=N_FACTORS, spikeslab_weights=True, ard_factors=True)
ent.set_train_options(
    iter        = 1000,
    convergence_mode = "fast",
    startELBO   = 1,
    seed        = 42,
    verbose     = False
)

ent.build()
ent.run()

# ── Extract results ───────────────────────────────────────────────────────────
model = ent.model

# Factor scores: cells x factors
Z = np.array(model.nodes["Z"].getExpectation())   # (n_cells, n_factors)
factor_df = pd.DataFrame(Z, index=shared, columns=[f"Factor{i+1}" for i in range(N_FACTORS)])
factor_df = factor_df.join(meta[["cell_type", "condition"]], how="left")
factor_df["condition_label"] = factor_df["condition"].map({"WT2": "WT", "KO3": "KO"}).fillna(factor_df["condition"])
factor_df.to_csv(os.path.join(OUT_DIR, "MOFA_factor_scores.csv"))
print(f"Factor scores saved: MOFA_factor_scores.csv")

# Variance explained per factor per view
r2 = model.calculate_variance_explained()
r2_df = pd.DataFrame(
    {view: r2[view][0] for view in ["RNA", "ATAC"]},
    index=[f"Factor{i+1}" for i in range(N_FACTORS)]
)
r2_df.to_csv(os.path.join(OUT_DIR, "MOFA_variance_explained.csv"))
print(f"Variance explained saved: MOFA_variance_explained.csv")

# Top feature weights per factor
W_rna  = np.array(model.nodes["W"].getExpectation()[0])   # (n_genes, n_factors)
W_atac = np.array(model.nodes["W"].getExpectation()[1])   # (50, n_factors)

weights_rows = []
for f_idx in range(N_FACTORS):
    factor_name = f"Factor{f_idx+1}"
    # Top 20 RNA features
    top_idx = np.argsort(np.abs(W_rna[:, f_idx]))[::-1][:20]
    for rank, idx in enumerate(top_idx):
        weights_rows.append({
            "factor": factor_name, "view": "RNA",
            "feature": adata_rna.var_names[idx],
            "weight": W_rna[idx, f_idx], "rank": rank + 1
        })

weights_df = pd.DataFrame(weights_rows)
weights_df.to_csv(os.path.join(OUT_DIR, "MOFA_top_weights.csv"), index=False)
print(f"Top weights saved: MOFA_top_weights.csv")

# ── Plot 1: Variance explained heatmap ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(r2_df.T * 100, annot=True, fmt=".1f", cmap="YlOrRd",
            ax=ax, cbar_kws={"label": "% variance explained"})
ax.set_title("MOFA+ — Variance explained per factor and view")
ax.set_xlabel("Factor")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "MOFA_variance_explained.pdf"), bbox_inches="tight")
plt.close()
print("Saved: MOFA_variance_explained.pdf")

# ── Plot 2: UMAP on MOFA factor scores ───────────────────────────────────────
print("\nComputing UMAP on MOFA factors...")
adata_mofa = anndata.AnnData(
    X   = Z.astype(np.float32),
    obs = factor_df[["cell_type", "condition_label"]].copy()
)
sc.pp.neighbors(adata_mofa, n_neighbors=30, use_rep="X")
sc.tl.umap(adata_mofa, min_dist=0.3)

umap = adata_mofa.obsm["X_umap"]
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, (col, palette, title) in zip(axes, [
    ("cell_type",        CELLTYPE_COLORS,  "Cell type"),
    ("condition_label",  CONDITION_COLORS, "Condition (KO vs WT)")
]):
    cats = adata_mofa.obs[col].dropna().unique()
    for cat in cats:
        mask = adata_mofa.obs[col] == cat
        ax.scatter(umap[mask, 0], umap[mask, 1],
                   c=palette.get(str(cat), "#999"), s=5,
                   alpha=0.6, label=str(cat), rasterized=True)
    ax.set_title(f"MOFA+ UMAP — {title}")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=3, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "MOFA_UMAP.pdf"), bbox_inches="tight", dpi=200)
plt.close()
print("Saved: MOFA_UMAP.pdf")

# ── Plot 3: Factor scores KO vs WT per cell type (violin) ────────────────────
factor_cols = [c for c in factor_df.columns if c.startswith("Factor")]

# Find factors most associated with KO vs WT (largest mean difference)
ko_means = factor_df[factor_df["condition_label"] == "KO"][factor_cols].mean()
wt_means = factor_df[factor_df["condition_label"] == "WT"][factor_cols].mean()
ko_wt_diff = (ko_means - wt_means).abs().sort_values(ascending=False)
top_factors = ko_wt_diff.index[:6].tolist()

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for ax, factor in zip(axes, top_factors):
    data = []
    labels = []
    for ct in ["LHS", "LASPs", "BMyo", "BL"]:
        for cond, color in [("WT", "#2196F3"), ("KO", "#F44336")]:
            vals = factor_df[
                (factor_df["cell_type"] == ct) & (factor_df["condition_label"] == cond)
            ][factor].dropna().values
            if len(vals) > 0:
                data.append(vals)
                labels.append(f"{ct}\n{cond}")

    parts = ax.violinplot(data, showmedians=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
    ax.set_title(factor)
    ax.set_ylabel("Factor score")

plt.suptitle("MOFA+ factors most different between KO and WT", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "MOFA_KO_vs_WT_factors.pdf"), bbox_inches="tight")
plt.close()
print("Saved: MOFA_KO_vs_WT_factors.pdf")

# ── Plot 4: Top RNA features per top factor (barplot of weights) ──────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for ax, factor in zip(axes, top_factors):
    f_idx = int(factor.replace("Factor", "")) - 1
    w = W_rna[:, f_idx]
    top20 = np.argsort(np.abs(w))[::-1][:20]
    genes = [adata_rna.var_names[i] for i in top20]
    vals  = [w[i] for i in top20]
    colors = ["#F44336" if v < 0 else "#2196F3" for v in vals]
    ax.barh(genes[::-1], vals[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_title(f"{factor} — top RNA weights")
    ax.set_xlabel("Weight")
    ax.tick_params(axis="y", labelsize=7)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "MOFA_top_RNA_weights.pdf"), bbox_inches="tight")
plt.close()
print("Saved: MOFA_top_RNA_weights.pdf")

print(f"\nAll done. Outputs in: {OUT_DIR}")
print("Key files:")
print("  MOFA_factor_scores.csv    — factor score per cell")
print("  MOFA_variance_explained.csv — R² per factor per view")
print("  MOFA_top_weights.csv      — top genes per factor")
print("  MOFA_UMAP.pdf             — UMAP on MOFA factors")
print("  MOFA_KO_vs_WT_factors.pdf — KO vs WT violin plots")
print("  MOFA_top_RNA_weights.pdf  — top genes driving each factor")
