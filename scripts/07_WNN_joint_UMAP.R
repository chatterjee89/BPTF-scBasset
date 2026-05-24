# 07_WNN_joint_UMAP.R
# Weighted Nearest Neighbour (WNN) joint UMAP combining:
#   - RNA PCA embeddings  (from existing Seurat object)
#   - scBasset cell embeddings (32-dim, sequence-based)
# Run after 04_embeddings.py has produced scbasset_embeddings.csv

library(Seurat)
library(ggplot2)
library(patchwork)

# ── Paths ─────────────────────────────────────────────────────────────────────
RDS_PATH        <- "~/Documents/Coding/WT2_KO3_epithelia.Rds"
EMBEDDINGS_CSV  <- "~/Documents/Coding/BPTF_scDataset/scBasset/outputs/embeddings/scbasset_cell_embeddings.csv"
OUT_DIR         <- "~/Documents/Coding/BPTF_scDataset/scBasset/outputs/WNN"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

# ── Condition colour palette (consistent with other plots) ────────────────────
CONDITION_COLORS <- c("WT" = "#2196F3", "KO" = "#F44336")
CELLTYPE_COLORS  <- c("LHS" = "#4CAF50", "LASPs" = "#FF9800",
                      "BMyo" = "#9C27B0", "BL" = "#00BCD4")

# ── Load Seurat object ────────────────────────────────────────────────────────
message("Loading Seurat object...")
seurat_obj <- readRDS(RDS_PATH)
message(sprintf("  %d cells, %d genes", ncol(seurat_obj), nrow(seurat_obj)))

# ── Normalise and run PCA on RNA if not already done ─────────────────────────
if (!"pca" %in% names(seurat_obj@reductions)) {
  message("Running RNA normalisation and PCA...")
  seurat_obj <- NormalizeData(seurat_obj)
  seurat_obj <- FindVariableFeatures(seurat_obj, nfeatures = 3000)
  seurat_obj <- ScaleData(seurat_obj)
  seurat_obj <- RunPCA(seurat_obj, npcs = 50)
}

# ── Load scBasset embeddings ──────────────────────────────────────────────────
message("Loading scBasset embeddings...")
emb <- read.csv(EMBEDDINGS_CSV, row.names = 1)
# emb: rows = barcodes, cols = 32 embedding dimensions

# Keep only cells present in both the Seurat object and the embeddings
shared_cells <- intersect(colnames(seurat_obj), rownames(emb))
message(sprintf("  Shared cells: %d", length(shared_cells)))

seurat_sub <- seurat_obj[, shared_cells]
emb_mat    <- as.matrix(emb[shared_cells, ])

# ── Add scBasset as a new DimReduc ────────────────────────────────────────────
seurat_sub[["scbasset"]] <- CreateDimReducObject(
  embeddings = emb_mat,
  key        = "scbasset_",
  assay      = DefaultAssay(seurat_sub)
)

# ── WNN: weight RNA PCA and scBasset embeddings per cell ─────────────────────
message("Running WNN...")
seurat_sub <- FindMultiModalNeighbors(
  seurat_sub,
  reduction.list = list("pca", "scbasset"),
  dims.list      = list(1:30, 1:32),
  modality.weight.name = "RNA.weight"
)

# ── UMAP on WNN graph ─────────────────────────────────────────────────────────
message("Running UMAP...")
seurat_sub <- RunUMAP(
  seurat_sub,
  nn.name    = "weighted.nn",
  reduction.name = "wnn.umap",
  reduction.key = "wnnUMAP_"
)

# ── Recode condition labels ───────────────────────────────────────────────────
seurat_sub$condition <- ifelse(seurat_sub$orig.ident == "WT2", "WT", "KO")

# ── Plot 1: WNN UMAP coloured by cell type ───────────────────────────────────
p1 <- DimPlot(seurat_sub, reduction = "wnn.umap", group.by = "cell_type",
              cols = CELLTYPE_COLORS, pt.size = 0.6, raster = FALSE) +
  ggtitle("WNN UMAP — cell type") +
  theme_classic(base_size = 12) +
  theme(legend.position = "right")

# ── Plot 2: WNN UMAP coloured by condition ────────────────────────────────────
p2 <- DimPlot(seurat_sub, reduction = "wnn.umap", group.by = "condition",
              cols = CONDITION_COLORS, pt.size = 0.6, raster = FALSE) +
  ggtitle("WNN UMAP — KO vs WT") +
  theme_classic(base_size = 12) +
  theme(legend.position = "right")

# ── Plot 3: RNA-only UMAP for comparison ─────────────────────────────────────
seurat_sub <- RunUMAP(seurat_sub, reduction = "pca", dims = 1:30,
                      reduction.name = "rna.umap", reduction.key = "rnaUMAP_")

p3 <- DimPlot(seurat_sub, reduction = "rna.umap", group.by = "cell_type",
              cols = CELLTYPE_COLORS, pt.size = 0.6, raster = FALSE) +
  ggtitle("RNA-only UMAP — cell type") +
  theme_classic(base_size = 12) +
  theme(legend.position = "right")

p4 <- DimPlot(seurat_sub, reduction = "rna.umap", group.by = "condition",
              cols = CONDITION_COLORS, pt.size = 0.6, raster = FALSE) +
  ggtitle("RNA-only UMAP — KO vs WT") +
  theme_classic(base_size = 12) +
  theme(legend.position = "right")

# ── Combine and save ──────────────────────────────────────────────────────────
combined <- (p1 | p2) / (p3 | p4)
ggsave(file.path(OUT_DIR, "WNN_vs_RNA_UMAP.pdf"),
       plot = combined, width = 14, height = 12)
message("Saved: WNN_vs_RNA_UMAP.pdf")

# ── Plot 4: per cell type, split by condition ─────────────────────────────────
pdf(file.path(OUT_DIR, "WNN_UMAP_split_KO_WT.pdf"), width = 10, height = 10)
for (ct in c("LHS", "LASPs", "BMyo", "BL")) {
  cells_ct <- WhichCells(seurat_sub, expression = cell_type == ct)
  p <- DimPlot(seurat_sub, reduction = "wnn.umap", cells.highlight = cells_ct,
               group.by = "condition", cols = CONDITION_COLORS,
               split.by = "condition", pt.size = 0.8, raster = FALSE) +
    ggtitle(ct) + theme_classic(base_size = 11)
  print(p)
}
dev.off()
message("Saved: WNN_UMAP_split_KO_WT.pdf")

# ── Plot 5: RNA modality weights (how much RNA vs scBasset per cell) ──────────
if ("RNA.weight" %in% colnames(seurat_sub@meta.data)) {
  p_weight <- FeaturePlot(seurat_sub, reduction = "wnn.umap",
                           features = "RNA.weight", pt.size = 0.6) +
    scale_color_gradient2(low = "#2196F3", mid = "white", high = "#F44336",
                          midpoint = 0.5) +
    ggtitle("WNN RNA weight\n(high = RNA-dominant, low = scBasset-dominant)") +
    theme_classic(base_size = 12)
  ggsave(file.path(OUT_DIR, "WNN_RNA_weights.pdf"),
         plot = p_weight, width = 7, height = 6)
  message("Saved: WNN_RNA_weights.pdf")
}

# ── Save updated Seurat object ────────────────────────────────────────────────
saveRDS(seurat_sub, file.path(OUT_DIR, "seurat_WNN.Rds"))
message("Saved: seurat_WNN.Rds")
message("\nDone. WNN UMAP complete.")
