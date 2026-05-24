"""
04_embeddings.py
Extract scBasset cell embeddings → UMAP → compare KO vs WT per cell type.
Run after 03_train_scbasset.py.
"""

import os
import numpy as np
import pandas as pd
import anndata
import h5py
import tensorflow as tf
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib as mpl

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.expanduser("~/Documents/Coding/BPTF_scDataset/scBasset/data")
MODEL_DIR  = os.path.expanduser("~/Documents/Coding/BPTF_scDataset/scBasset/models")
OUT_DIR    = os.path.expanduser("~/Documents/Coding/BPTF_scDataset/scBasset/outputs/embeddings")
os.makedirs(OUT_DIR, exist_ok=True)

ADATA_PATH = os.path.join(DATA_DIR, "atac_for_scbasset.h5ad")
SEQ_H5     = os.path.join(DATA_DIR, "sequences.h5")
MODEL_PATH = os.path.join(MODEL_DIR, "scbasset_model.h5")

# ── Colors matching your existing plots ───────────────────────────────────────
CONDITION_COLORS = {"WT": "#2196F3", "KO": "#F44336"}
CELLTYPE_COLORS  = {
    "LHS":   "#4CAF50",
    "LASPs": "#FF9800",
    "BMyo":  "#9C27B0",
    "BL":    "#00BCD4",
}

# ── Load model and data ───────────────────────────────────────────────────────
print("Loading trained model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Loading sequences and AnnData...")
adata = anndata.read_h5ad(ADATA_PATH)
with h5py.File(SEQ_H5, "r") as f:
    sequences = f["sequences"][:]

print(f"  {sequences.shape[0]} peaks, {adata.n_obs} cells")

# ── Extract cell embeddings ───────────────────────────────────────────────────
# The cell embedding is the weight matrix of the final Dense layer (cell_embedding → output)
# Each column = embedding for one cell
output_layer = model.get_layer("accessibility")
cell_weights = output_layer.get_weights()[0]    # shape: (32, n_cells)
cell_embeddings = cell_weights.T                # shape: (n_cells, 32)

print(f"Cell embeddings shape: {cell_embeddings.shape}")

# Attach to AnnData
adata_cells = anndata.AnnData(
    X=cell_embeddings,
    obs=adata.obs.copy()
)
adata_cells.obsm["X_scbasset"] = cell_embeddings

# ── UMAP on cell embeddings ───────────────────────────────────────────────────
print("\nComputing UMAP...")
sc.pp.neighbors(adata_cells, use_rep="X_scbasset", n_neighbors=30, metric="cosine")
sc.tl.umap(adata_cells, min_dist=0.3)

# Save UMAP coordinates back to main adata
adata.obsm["X_scbasset"] = cell_embeddings
adata.obsm["X_umap_scbasset"] = adata_cells.obsm["X_umap"]

# Save updated adata
adata.write_h5ad(os.path.join(DATA_DIR, "atac_for_scbasset_with_embeddings.h5ad"))

# ── Detect cell type and condition columns ────────────────────────────────────
ct_col = next(
    (c for c in adata.obs.columns if "cell" in c.lower() or "type" in c.lower()), None
)
cond_col = "condition" if "condition" in adata.obs.columns else None
print(f"  Cell type column: {ct_col}")
print(f"  Condition column: {cond_col}")

# ── Plot 1: UMAP colored by cell type ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

umap = adata.obsm["X_umap_scbasset"]

for ax, (col, palette, title) in zip(axes, [
    (ct_col, CELLTYPE_COLORS, "Cell type"),
    (cond_col, CONDITION_COLORS, "Condition (KO vs WT)")
]):
    if col is None:
        ax.set_visible(False)
        continue
    cats = adata.obs[col].unique()
    for cat in cats:
        mask = adata.obs[col] == cat
        color = palette.get(str(cat), "#999999")
        ax.scatter(umap[mask, 0], umap[mask, 1],
                   c=color, s=4, alpha=0.6, label=str(cat), rasterized=True)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(f"scBasset embeddings — {title}")
    ax.legend(markerscale=3, bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "umap_scbasset_celltype_condition.pdf"),
            bbox_inches="tight", dpi=200)
plt.close()
print("Saved: umap_scbasset_celltype_condition.pdf")

# ── Plot 2: Side-by-side UMAP per cell type, split by condition ──────────────
if ct_col and cond_col:
    cell_types = sorted(adata.obs[ct_col].dropna().unique())
    fig, axes = plt.subplots(len(cell_types), 2, figsize=(10, 4 * len(cell_types)))

    for row_i, ct in enumerate(cell_types):
        for col_i, cond in enumerate(["WT", "KO"]):
            ax = axes[row_i, col_i]
            # Background: all cells
            ax.scatter(umap[:, 0], umap[:, 1], c="#EEEEEE", s=2, alpha=0.3, rasterized=True)
            # Foreground: this cell type + condition
            mask = (adata.obs[ct_col] == ct) & (adata.obs[cond_col] == cond)
            color = CONDITION_COLORS.get(cond, "#999999")
            ax.scatter(umap[mask, 0], umap[mask, 1],
                       c=color, s=6, alpha=0.7, rasterized=True)
            ax.set_title(f"{ct} — {cond} (n={mask.sum()})")
            ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle("scBasset embeddings by cell type and condition", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "umap_scbasset_split_KO_WT.pdf"),
                bbox_inches="tight", dpi=200)
    plt.close()
    print("Saved: umap_scbasset_split_KO_WT.pdf")

# ── Plot 3: Training curve ────────────────────────────────────────────────────
import json
log_path = os.path.expanduser(
    "~/Documents/Coding/BPTF_scDataset/scBasset/logs/training_history.json"
)
if os.path.exists(log_path):
    with open(log_path) as f:
        hist = json.load(f)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, title in zip(axes,
                                  ["loss", "auc"],
                                  ["Loss (binary cross-entropy)", "AUC"]):
        ax.plot(hist[metric], label="train")
        val_key = f"val_{metric}"
        if val_key in hist:
            ax.plot(hist[val_key], label="val")
        ax.set_xlabel("Epoch"); ax.set_ylabel(title)
        ax.set_title(title); ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "training_curves.pdf"), bbox_inches="tight")
    plt.close()
    print("Saved: training_curves.pdf")

print("\nDone. Next: run 05_motif_analysis.py")
