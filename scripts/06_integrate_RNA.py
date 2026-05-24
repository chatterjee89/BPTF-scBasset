"""
06_integrate_RNA.py
Three-part integration of scBasset results with RNA data:
  A) Cross-validate scBasset ISM motif importance against SCENIC+ eGRN TF-peak links
  B) Correlate ISM KO-WT delta with RNA differential TF expression
  C) Link high-ISM peaks to nearby DEGs via SCENIC+ peak-gene scores

Run after 05_motif_analysis.py.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# ── Paths ─────────────────────────────────────────────────────────────────────
SCBASSET_DIR  = os.path.expanduser("~/Documents/Coding/BPTF_scDataset/scBasset")
PYSCENIC_DIR  = os.path.expanduser("~/Documents/Coding/BPTF_scDataset/pySCENIC/output")
DGE_DIR       = os.path.expanduser("~/Documents/Coding/BPTF_scDataset/data/DGE")
OUT_DIR       = os.path.join(SCBASSET_DIR, "outputs", "rna_integration")
os.makedirs(OUT_DIR, exist_ok=True)

MOTIF_DIR     = os.path.join(SCBASSET_DIR, "outputs", "motifs")
DIFF_PEAKS    = os.path.join(MOTIF_DIR, "top_differential_peaks_summary.csv")
ISM_RESULTS   = os.path.join(MOTIF_DIR, "ism_results.npy")
EREGULONS     = os.path.join(PYSCENIC_DIR, "scenicplus", "eregulons.csv")
R2G           = os.path.join(PYSCENIC_DIR, "scenicplus", "r2g_adj.csv")
TF2G          = os.path.join(PYSCENIC_DIR, "scenicplus", "tf2g_adj.csv")

CELL_TYPES    = ["LHS", "LASPs", "BMyo", "BL"]
CONDITION_COLORS = {"WT": "#2196F3", "KO": "#F44336"}

# ─────────────────────────────────────────────────────────────────────────────
# PART A: Cross-validate scBasset ISM with SCENIC+ eGRN TF-peak links
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("PART A: scBasset ISM vs SCENIC+ eGRN cross-validation")
print("=" * 60)

# Load SCENIC+ peak-TF links (eregulons.csv: Region → TF)
print("Loading SCENIC+ eregulons...")
ereg = pd.read_csv(EREGULONS)
# Normalise peak format to match scBasset (chr1-start-end)
ereg["Region"] = ereg["Region"].str.replace(":", "-").str.replace("_", "-")
print(f"  {len(ereg)} TF-peak links, {ereg['TF'].nunique()} TFs")

# Load differential eRegulon scores per cell type
diff_ereg_frames = []
for ct in CELL_TYPES:
    path = os.path.join(PYSCENIC_DIR, "diff_eregulon", f"diff_eRegulon_{ct}_KOvWT.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["cell_type"] = ct
        diff_ereg_frames.append(df)
diff_ereg = pd.concat(diff_ereg_frames, ignore_index=True)

# Extract TF name from eRegulon string e.g. "Foxq1_extended_-_-_(230g)" → "Foxq1"
diff_ereg["TF"] = diff_ereg["eRegulon"].str.extract(r"^([^_]+)")
diff_ereg["direction"] = np.where(diff_ereg["mean_diff"] < 0, "KO_lower", "KO_higher")
print(f"  Differential eRegulons loaded: {len(diff_ereg)} rows")

# Load scBasset differential peak summary
if os.path.exists(DIFF_PEAKS):
    diff_peaks_df = pd.read_csv(DIFF_PEAKS)
    print(f"  scBasset differential peaks: {len(diff_peaks_df)} rows")

    # For each significant eRegulon (padj < 0.05), check whether its peaks
    # show matching directional change in scBasset accessibility
    print("\nCross-validating eRegulon directions with scBasset peak accessibility...")
    validation_rows = []
    sig_ereg = diff_ereg[diff_ereg["padj"] < 0.05].copy()

    for _, ereg_row in sig_ereg.iterrows():
        tf = ereg_row["TF"]
        ct = ereg_row["cell_type"]
        ereg_direction = ereg_row["direction"]

        # Get peaks linked to this TF in SCENIC+
        tf_peaks = ereg[ereg["TF"] == tf]["Region"].tolist()
        if not tf_peaks:
            continue

        # Check scBasset accessibility delta for these peaks
        matching = diff_peaks_df[
            (diff_peaks_df["cell_type"] == ct) &
            (diff_peaks_df["peak"].isin(tf_peaks))
        ]
        if matching.empty:
            continue

        mean_delta = matching["delta_KO_minus_WT"].mean()
        scbasset_direction = "KO_lower" if mean_delta < 0 else "KO_higher"
        concordant = (ereg_direction == scbasset_direction)

        validation_rows.append({
            "TF": tf,
            "cell_type": ct,
            "eregulon_direction": ereg_direction,
            "eregulon_log2FC": ereg_row["log2FC"],
            "eregulon_padj": ereg_row["padj"],
            "n_peaks_checked": len(matching),
            "scbasset_mean_delta": mean_delta,
            "scbasset_direction": scbasset_direction,
            "concordant": concordant,
        })

    if validation_rows:
        val_df = pd.DataFrame(validation_rows)
        val_df.to_csv(os.path.join(OUT_DIR, "eRegulon_scBasset_crossvalidation.csv"), index=False)
        pct_concordant = val_df["concordant"].mean() * 100
        print(f"  Concordance: {pct_concordant:.1f}% of eRegulons agree in direction")
        print(f"  Results saved: eRegulon_scBasset_crossvalidation.csv")

        # Plot: eRegulon log2FC vs scBasset peak delta, coloured by cell type
        fig, ax = plt.subplots(figsize=(8, 6))
        ct_palette = {"LHS": "#4CAF50", "LASPs": "#FF9800", "BMyo": "#9C27B0", "BL": "#00BCD4"}
        for ct, grp in val_df.groupby("cell_type"):
            ax.scatter(grp["eregulon_log2FC"], grp["scbasset_mean_delta"],
                       c=ct_palette.get(ct, "#999"), label=ct, alpha=0.7, s=60)
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        ax.axvline(0, color="black", linewidth=0.7, linestyle="--")
        ax.set_xlabel("SCENIC+ eRegulon log2FC (KO vs WT)")
        ax.set_ylabel("scBasset mean peak delta (KO − WT)")
        ax.set_title("eRegulon activity vs. scBasset peak accessibility")
        ax.legend(frameon=False)
        plt.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "crossvalidation_scatter.pdf"), bbox_inches="tight")
        plt.close()
        print("  Saved: crossvalidation_scatter.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# PART B: ISM KO-WT delta vs differential TF expression (RNA)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PART B: ISM motif importance delta vs RNA TF expression")
print("=" * 60)

# Load RNA DEGs for all cell types
dge_frames = []
for ct in CELL_TYPES:
    path = os.path.join(DGE_DIR, f"{ct}_KO3vWT2_DGE.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["cell_type"] = ct
        dge_frames.append(df)
dge = pd.concat(dge_frames, ignore_index=True)
print(f"  Loaded DEGs: {len(dge)} rows across {dge['cell_type'].nunique()} cell types")

# Get list of TFs from SCENIC+ eRegulons
tf_list = ereg["TF"].unique().tolist()
print(f"  SCENIC+ TFs to check: {len(tf_list)}")

# Filter DEGs to TFs only
dge_tfs = dge[dge["gene"].isin(tf_list)].copy()
print(f"  DEG entries that are TFs: {len(dge_tfs)}")

# Merge with differential eRegulon (scBasset-side proxy = eRegulon motif direction)
merged = dge_tfs.merge(
    diff_ereg[["TF", "cell_type", "log2FC", "mean_diff", "padj"]],
    left_on=["gene", "cell_type"],
    right_on=["TF", "cell_type"],
    how="inner"
)
merged.rename(columns={
    "avg_log2FC": "RNA_log2FC",
    "log2FC": "eRegulon_log2FC",
    "mean_diff": "eRegulon_mean_diff",
    "padj": "eRegulon_padj",
    "p_val_adj": "RNA_FDR"
}, inplace=True)

if not merged.empty:
    merged.to_csv(os.path.join(OUT_DIR, "TF_RNA_vs_eRegulon_motif.csv"), index=False)
    print(f"  Merged TF table: {len(merged)} rows saved")

    # Classify each TF into one of four mechanistic categories
    merged["RNA_sig"]    = merged["RNA_FDR"] < 0.05
    merged["eReg_sig"]   = merged["eRegulon_padj"] < 0.05
    merged["mechanism"] = "not_significant"
    merged.loc[merged["RNA_sig"] & ~merged["eReg_sig"],  "mechanism"] = "expression_only"
    merged.loc[~merged["RNA_sig"] & merged["eReg_sig"],  "mechanism"] = "accessibility_only"
    merged.loc[merged["RNA_sig"] & merged["eReg_sig"],   "mechanism"] = "both"

    mech_colors = {
        "both":             "#E91E63",
        "expression_only":  "#FF9800",
        "accessibility_only": "#2196F3",
        "not_significant":  "#CCCCCC",
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    for mech, grp in merged.groupby("mechanism"):
        ax.scatter(grp["RNA_log2FC"], grp["eRegulon_log2FC"],
                   c=mech_colors.get(mech, "#999"), label=mech,
                   alpha=0.75, s=60, zorder=3 if mech == "both" else 2)
        # Label the "both" TFs
        if mech == "both":
            for _, row in grp.iterrows():
                ax.annotate(row["gene"], (row["RNA_log2FC"], row["eRegulon_log2FC"]),
                            fontsize=6, alpha=0.8,
                            xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.axvline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_xlabel("RNA log2FC (KO vs WT)")
    ax.set_ylabel("eRegulon activity log2FC (KO vs WT)")
    ax.set_title("TF expression vs chromatin accessibility change\n(KO vs WT)")
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "TF_RNA_vs_motif_quadrant.pdf"), bbox_inches="tight")
    plt.close()
    print("  Saved: TF_RNA_vs_motif_quadrant.pdf")

    # Print the "accessibility_only" TFs — most mechanistically interesting
    acc_only = merged[merged["mechanism"] == "accessibility_only"].sort_values("eRegulon_log2FC")
    if not acc_only.empty:
        print("\n  TFs with changed CHROMATIN ACCESS but UNCHANGED RNA expression:")
        print("  (These require BPTF for site access, not for transcription)")
        print(acc_only[["gene", "cell_type", "RNA_log2FC", "RNA_FDR",
                         "eRegulon_log2FC", "eRegulon_padj"]].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# PART C: Link high-ISM peaks to nearby DEGs via SCENIC+ peak-gene scores
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PART C: High-ISM peaks → nearby DEGs")
print("=" * 60)

print("Loading SCENIC+ peak-gene links...")
r2g = pd.read_csv(R2G)
r2g["Region"] = r2g["Region"].str.replace(":", "-").str.replace("_", "-") if "Region" in r2g.columns else r2g.iloc[:, 0]
print(f"  {len(r2g)} peak-gene links")

if os.path.exists(DIFF_PEAKS):
    # Take top 20% most differential peaks (largest |KO - WT| delta)
    diff_peaks_df["abs_delta"] = diff_peaks_df["delta_KO_minus_WT"].abs()
    threshold = diff_peaks_df["abs_delta"].quantile(0.80)
    high_ism_peaks = diff_peaks_df[diff_peaks_df["abs_delta"] >= threshold].copy()
    print(f"  High-ISM peaks (top 20% delta): {len(high_ism_peaks)}")

    # Link peaks → genes via SCENIC+ r2g
    linked = high_ism_peaks.merge(
        r2g[["Region", "Gene", "R2G_importance", "R2G_rho"]],
        left_on="peak", right_on="Region", how="left"
    ).dropna(subset=["Gene"])
    print(f"  High-ISM peaks with SCENIC+ gene links: {len(linked)}")

    # Add RNA DEG info
    linked = linked.merge(
        dge[["gene", "cell_type", "avg_log2FC", "p_val_adj", "FDR"]],
        left_on=["Gene", "cell_type"], right_on=["gene", "cell_type"],
        how="left"
    )
    linked["is_DEG"] = linked["FDR"] < 0.05
    linked.to_csv(os.path.join(OUT_DIR, "high_ISM_peaks_to_DEGs.csv"), index=False)
    print(f"  Saved: high_ISM_peaks_to_DEGs.csv")

    # Summary: how many high-ISM peaks link to a DEG?
    if "is_DEG" in linked.columns:
        linked_to_deg = linked.groupby("cell_type")["is_DEG"].agg(["sum", "count"])
        linked_to_deg.columns = ["n_linked_to_DEG", "n_total_links"]
        linked_to_deg["pct"] = (linked_to_deg["n_linked_to_DEG"] / linked_to_deg["n_total_links"] * 100).round(1)
        print("\n  High-ISM peaks linked to DEGs per cell type:")
        print(linked_to_deg.to_string())

    # Plot: for each cell type, top 15 DEGs linked to high-ISM peaks
    for ct in CELL_TYPES:
        ct_linked = linked[
            (linked["cell_type"] == ct) & (linked["is_DEG"] == True)
        ].sort_values("abs_delta", ascending=False).drop_duplicates("Gene").head(15)

        if ct_linked.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#F44336" if v < 0 else "#2196F3" for v in ct_linked["avg_log2FC"]]
        bars = ax.barh(ct_linked["Gene"], ct_linked["avg_log2FC"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("RNA log2FC (KO vs WT)")
        ax.set_title(f"{ct} — DEGs linked to high-ISM peaks")
        plt.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"high_ISM_DEGs_{ct}.pdf"), bbox_inches="tight")
        plt.close()

    print("\n  Saved per-cell-type DEG bar plots")

print("\nAll done — Part A, B, C complete.")
print(f"Outputs in: {OUT_DIR}")
