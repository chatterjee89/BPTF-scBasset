"""
05_motif_analysis.py
In silico mutagenesis (ISM) on peaks of interest → TF motif importance per cell type.
Identifies which TF motifs drive accessibility differences in BPTF KO vs WT.
Run after 04_embeddings.py.
"""

import os
import numpy as np
import pandas as pd
import anndata
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = os.path.expanduser("~/Documents/Coding/BPTF_scDataset/scBasset/data")
MODEL_DIR = os.path.expanduser("~/Documents/Coding/BPTF_scDataset/scBasset/models")
OUT_DIR   = os.path.expanduser("~/Documents/Coding/BPTF_scDataset/scBasset/outputs/motifs")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "scbasset_model.h5")
ADATA_PATH = os.path.join(DATA_DIR, "atac_for_scbasset_with_embeddings.h5ad")
SEQ_H5     = os.path.join(DATA_DIR, "sequences.h5")

NUCLEOTIDES = ["A", "C", "G", "T"]

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading model and data...")
model = tf.keras.models.load_model(MODEL_PATH)

adata = anndata.read_h5ad(ADATA_PATH)
with h5py.File(SEQ_H5, "r") as f:
    sequences = f["sequences"][:]
    peak_ids  = list(f["peak_ids"][:])

# Detect cell type and condition columns
ct_col   = next((c for c in adata.obs.columns if "cell" in c.lower() or "type" in c.lower()), None)
cond_col = "condition" if "condition" in adata.obs.columns else None

# Build group index: which cells belong to each (cell_type, condition) group
def get_group_indices(adata, ct_col, cond_col):
    groups = {}
    if ct_col:
        for ct in adata.obs[ct_col].dropna().unique():
            mask_ct = adata.obs[ct_col] == ct
            if cond_col:
                for cond in ["WT", "KO"]:
                    mask = mask_ct & (adata.obs[cond_col] == cond)
                    if mask.sum() > 0:
                        groups[f"{ct}_{cond}"] = np.where(mask)[0]
            else:
                groups[ct] = np.where(mask_ct)[0]
    return groups

groups = get_group_indices(adata, ct_col, cond_col)
print(f"Groups: {list(groups.keys())}")

# ── ISM: In Silico Mutagenesis ────────────────────────────────────────────────
def run_ism(model, sequence, cell_indices):
    """
    For each position in sequence, substitute each nucleotide and measure
    the change in mean predicted accessibility for the given cells.
    Returns: (seq_len, 4) importance scores.
    """
    seq_len = sequence.shape[0]
    ref_pred = model.predict(sequence[np.newaxis], verbose=0)[0, cell_indices]  # (n_cells,)
    ref_mean = ref_pred.mean()

    scores = np.zeros((seq_len, 4), dtype=np.float32)
    for pos in range(seq_len):
        for nuc_idx in range(4):
            if sequence[pos, nuc_idx] == 1:
                scores[pos, nuc_idx] = 0.0   # reference base
                continue
            mut_seq = sequence.copy()
            mut_seq[pos] = 0.0
            mut_seq[pos, nuc_idx] = 1.0
            mut_pred = model.predict(mut_seq[np.newaxis], verbose=0)[0, cell_indices]
            scores[pos, nuc_idx] = mut_pred.mean() - ref_mean  # positive = gain, negative = loss

    return scores

def plot_sequence_logo(ism_scores, peak_id, group_name, out_path, window=(600, 744)):
    """Plot ISM score logo for a central window of the peak."""
    start, end = window
    scores_window = ism_scores[start:end]   # (window_len, 4)

    fig, ax = plt.subplots(figsize=(20, 3))
    x = np.arange(scores_window.shape[0])
    colors = {"A": "#2ecc71", "C": "#3498db", "G": "#f39c12", "T": "#e74c3c"}

    for nuc_idx, nuc in enumerate(NUCLEOTIDES):
        ax.bar(x, scores_window[:, nuc_idx],
               color=colors[nuc], label=nuc, alpha=0.8, width=1.0)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Position in 1344bp window")
    ax.set_ylabel("ISM score\n(accessibility change)")
    ax.set_title(f"ISM — {peak_id} — {group_name}")
    ax.legend(loc="upper right", ncol=4, frameon=False)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()

# ── Select peaks for ISM ──────────────────────────────────────────────────────
# Strategy: pick top differentially accessible peaks (most variable across conditions)
import scipy.sparse

print("\nSelecting peaks for ISM analysis...")
if scipy.sparse.issparse(adata.X):
    mat = np.array(adata.X.todense())
else:
    mat = adata.X.copy()

# Compute mean accessibility per cell type x condition group
group_means = {}
for grp_name, cell_idx in groups.items():
    group_means[grp_name] = mat[cell_idx].mean(axis=0)   # (n_peaks,)

group_means_df = pd.DataFrame(group_means, index=adata.var_names)

# For each cell type, find peaks most different between KO and WT
CELL_TYPES = sorted({g.split("_")[0] for g in groups if "_" in g})
top_peaks_per_ct = {}
for ct in CELL_TYPES:
    ko_col = f"{ct}_KO"
    wt_col = f"{ct}_WT"
    if ko_col in group_means_df.columns and wt_col in group_means_df.columns:
        diff = (group_means_df[ko_col] - group_means_df[wt_col]).abs()
        top_peaks_per_ct[ct] = diff.nlargest(10).index.tolist()

print(f"Top differential peaks per cell type:")
for ct, peaks_list in top_peaks_per_ct.items():
    print(f"  {ct}: {len(peaks_list)} peaks selected for ISM")

# ── Run ISM on top peaks ──────────────────────────────────────────────────────
# Limit to a manageable number of peaks for runtime
MAX_PEAKS_PER_CT = 5   # ISM is slow on CPU; increase if using GPU

ism_results = {}  # peak_id → {group → ism_scores}

for ct in CELL_TYPES:
    if ct not in top_peaks_per_ct:
        continue
    selected_peaks = top_peaks_per_ct[ct][:MAX_PEAKS_PER_CT]

    for peak_id in tqdm(selected_peaks, desc=f"ISM {ct}"):
        if peak_id not in peak_ids:
            continue
        peak_seq_idx = peak_ids.index(peak_id)
        seq = sequences[peak_seq_idx]   # (1344, 4)

        ism_results[peak_id] = {}
        for grp_name in [f"{ct}_WT", f"{ct}_KO"]:
            if grp_name not in groups:
                continue
            cell_idx = groups[grp_name]
            scores = run_ism(model, seq, cell_idx)
            ism_results[peak_id][grp_name] = scores

            # Plot logo
            out_path = os.path.join(OUT_DIR, f"ism_{peak_id.replace(':', '-')}_{grp_name}.pdf")
            plot_sequence_logo(scores, peak_id, grp_name, out_path)

# ── KO vs WT ISM difference ───────────────────────────────────────────────────
print("\nPlotting KO vs WT ISM differences...")
for ct in CELL_TYPES:
    selected_peaks = top_peaks_per_ct.get(ct, [])[:MAX_PEAKS_PER_CT]
    for peak_id in selected_peaks:
        wt_key = f"{ct}_WT"
        ko_key = f"{ct}_KO"
        if peak_id not in ism_results:
            continue
        if wt_key not in ism_results[peak_id] or ko_key not in ism_results[peak_id]:
            continue

        diff_scores = ism_results[peak_id][ko_key] - ism_results[peak_id][wt_key]

        fig, ax = plt.subplots(figsize=(20, 3))
        x = np.arange(diff_scores.shape[0])
        colors = {"A": "#2ecc71", "C": "#3498db", "G": "#f39c12", "T": "#e74c3c"}
        for nuc_idx, nuc in enumerate(NUCLEOTIDES):
            ax.bar(x, diff_scores[:, nuc_idx], color=colors[nuc], label=nuc, alpha=0.8, width=1.0)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Position")
        ax.set_ylabel("ISM KO − WT")
        ax.set_title(f"ISM difference (KO − WT) — {peak_id} — {ct}")
        ax.legend(loc="upper right", ncol=4, frameon=False)
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"ism_diff_{peak_id.replace(':', '-')}_{ct}_KOvsWT.pdf")
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close()

# ── Save ISM scores as numpy archives ────────────────────────────────────────
np.save(os.path.join(OUT_DIR, "ism_results.npy"), ism_results)
print(f"\nISM results saved to {OUT_DIR}")

# ── Summary table of top differential peaks ───────────────────────────────────
summary_rows = []
for ct in CELL_TYPES:
    ko_col, wt_col = f"{ct}_KO", f"{ct}_WT"
    if ko_col in group_means_df.columns and wt_col in group_means_df.columns:
        for peak_id in top_peaks_per_ct.get(ct, []):
            summary_rows.append({
                "cell_type": ct,
                "peak": peak_id,
                "mean_accessibility_WT": group_means_df.loc[peak_id, wt_col],
                "mean_accessibility_KO": group_means_df.loc[peak_id, ko_col],
                "delta_KO_minus_WT": group_means_df.loc[peak_id, ko_col] - group_means_df.loc[peak_id, wt_col],
            })

summary_df = pd.DataFrame(summary_rows).sort_values(["cell_type", "delta_KO_minus_WT"])
summary_df.to_csv(os.path.join(OUT_DIR, "top_differential_peaks_summary.csv"), index=False)
print(f"Saved: top_differential_peaks_summary.csv")

print("\nAll done.")
print("Key outputs:")
print(f"  {OUT_DIR}/  — ISM logos and KO vs WT difference plots")
print(f"  {OUT_DIR}/top_differential_peaks_summary.csv")
print(f"  {OUT_DIR}/ism_results.npy")
