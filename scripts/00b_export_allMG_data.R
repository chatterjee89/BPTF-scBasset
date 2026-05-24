library(Seurat)
library(Signac)
library(Matrix)

RDS_PATH <- "~/Documents/Coding/allMG_BptfKOvWT_inclATAC.Rds"
OUT_DIR  <- "~/Documents/Coding/BPTF_scDataset/scBasset/data/allMG"

dir.create(file.path(OUT_DIR, "ATAC"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(OUT_DIR, "metadata"), recursive = TRUE, showWarnings = FALSE)

# ── Load ──────────────────────────────────────────────────────────────────────
cat("Loading full mammary gland object (9.3 GB — may take a few minutes)...\n")
obj <- readRDS(RDS_PATH)
cat(sprintf("Loaded: %d cells, %d assays\n", ncol(obj), length(Assays(obj))))

# ── Inspect cell types and conditions ─────────────────────────────────────────
cat("\nIdent levels:\n"); print(levels(Idents(obj)))
cat("\norig.ident counts:\n"); print(table(obj@meta.data$orig.ident))
cat("\nMeta.data columns:", colnames(obj@meta.data), "\n")

# Detect cell type column (may vary across objects)
ct_candidates <- c("cell_type", "celltype", "CellType", "predicted.id",
                   "annotation", "Annotation", "seurat_clusters")
ct_col <- ct_candidates[ct_candidates %in% colnames(obj@meta.data)][1]
if (is.na(ct_col)) {
  cat("No cell_type column found — using Idents()\n")
  obj@meta.data$cell_type <- as.character(Idents(obj))
  ct_col <- "cell_type"
}
cat(sprintf("\nUsing '%s' as cell type column:\n", ct_col))
print(table(obj@meta.data[[ct_col]]))

# Standardise: rename Mixed-lineage -> BL to match epithelial subset
obj@meta.data$cell_type <- as.character(obj@meta.data[[ct_col]])
obj@meta.data$cell_type[obj@meta.data$cell_type == "Mixed-lineage"] <- "BL"

# Detect condition from orig.ident
obj@meta.data$condition <- dplyr::case_when(
  grepl("WT",  obj@meta.data$orig.ident, ignore.case = TRUE) ~ "WT",
  grepl("KO",  obj@meta.data$orig.ident, ignore.case = TRUE) ~ "KO",
  TRUE ~ obj@meta.data$orig.ident
)
cat("\nCondition counts:\n"); print(table(obj@meta.data$condition))

# ── ATAC cells ────────────────────────────────────────────────────────────────
# Identify ATAC cells: those with ATAC assay data (peaks assay)
if ("peaks" %in% Assays(obj)) {
  DefaultAssay(obj) <- "peaks"
  atac_cells <- colnames(obj)[colSums(GetAssayData(obj, assay = "peaks",
                                                    layer = "counts")) > 0]
} else if ("ATAC" %in% Assays(obj)) {
  DefaultAssay(obj) <- "ATAC"
  atac_cells <- colnames(obj)[colSums(GetAssayData(obj, assay = "ATAC",
                                                    layer = "counts")) > 0]
} else {
  stop("No peaks or ATAC assay found in object.")
}

# Alternatively filter by orig.ident containing "ATAC" if multi-modal is stored that way
atac_idents <- grep("ATAC", obj@meta.data$orig.ident, value = TRUE, ignore.case = TRUE)
if (length(atac_idents) > 100) {
  atac_cells <- rownames(obj@meta.data)[obj@meta.data$orig.ident %in% unique(atac_idents)]
  cat(sprintf("\nATAC cells (from orig.ident): %d\n", length(atac_cells)))
} else {
  cat(sprintf("\nATAC cells (from assay): %d\n", length(atac_cells)))
}

atac_obj <- subset(obj, cells = atac_cells)
cat("Cell type breakdown (ATAC):\n"); print(table(atac_obj@meta.data$cell_type))
cat("Condition breakdown (ATAC):\n"); print(table(atac_obj@meta.data$condition))

# ── Export peak matrix ─────────────────────────────────────────────────────────
peak_assay <- intersect(c("peaks", "ATAC"), Assays(atac_obj))[1]
DefaultAssay(atac_obj) <- peak_assay
peak_mat     <- GetAssayData(atac_obj, assay = peak_assay, layer = "counts")
peak_mat_bin <- peak_mat; peak_mat_bin@x <- pmin(peak_mat_bin@x, 1)

writeMM(peak_mat_bin, file.path(OUT_DIR, "ATAC", "peak_counts_binary.mtx"))
writeMM(peak_mat,     file.path(OUT_DIR, "ATAC", "peak_counts_raw.mtx"))
writeLines(rownames(peak_mat), file.path(OUT_DIR, "ATAC", "peaks.txt"))
writeLines(colnames(peak_mat), file.path(OUT_DIR, "ATAC", "barcodes.txt"))
cat(sprintf("Peak matrix exported: %d peaks x %d cells\n",
            nrow(peak_mat), ncol(peak_mat)))

# ── Export peak BED ────────────────────────────────────────────────────────────
peaks_df <- data.frame(peak_id = rownames(peak_mat), stringsAsFactors = FALSE)
peaks_df$chr   <- sub("[-:].*", "", peaks_df$peak_id)
peaks_df$start <- as.integer(sub("^[^-:]+[-:]([0-9]+)[-:][0-9]+$", "\\1", peaks_df$peak_id))
peaks_df$end   <- as.integer(sub("^.*[-:]([0-9]+)$", "\\1", peaks_df$peak_id))
peaks_df       <- peaks_df[grepl("^chr([0-9]+|[XYM])$", peaks_df$chr), ]

write.table(peaks_df[, c("chr", "start", "end", "peak_id")],
            file.path(OUT_DIR, "ATAC", "peaks.bed"),
            sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)
cat(sprintf("Peak BED exported: %d standard-chromosome peaks\n", nrow(peaks_df)))

# ── Export ATAC metadata ───────────────────────────────────────────────────────
atac_meta <- atac_obj@meta.data[, intersect(
  c("orig.ident", "cell_type", "condition",
    "nCount_peaks", "nFeature_peaks", "nCount_ATAC", "nFeature_ATAC",
    "TSS.enrichment", "nucleosome_signal", "pct_reads_in_peaks"),
  colnames(atac_obj@meta.data)
)]
atac_meta$barcode <- rownames(atac_meta)
write.csv(atac_meta, file.path(OUT_DIR, "ATAC", "metadata.csv"), row.names = FALSE)

# ── Combined metadata ──────────────────────────────────────────────────────────
combined_meta <- data.frame(
  barcode   = rownames(atac_meta),
  cell_type = atac_meta$cell_type,
  condition = atac_meta$condition,
  orig.ident = atac_meta$orig.ident,
  stringsAsFactors = FALSE
)
write.csv(combined_meta,
          file.path(OUT_DIR, "metadata", "combined_metadata.csv"),
          row.names = FALSE)

cat("\n=== Export complete ===\n")
cat(sprintf("Output: %s\n", OUT_DIR))
cat("\nCell type x condition breakdown:\n")
print(table(combined_meta$cell_type, combined_meta$condition))
cat("\nDone. Next: run 01b_prepare_allMG_data.py\n")
