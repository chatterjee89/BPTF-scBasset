"""
03_train_scbasset.py
Train scBasset CNN on BPTF ATAC data.
Expects sequences.h5 and atac_for_scbasset.h5ad from previous steps.
Runtime: ~2-4 hours on CPU; ~30-60 min on GPU.
"""

import os
import json
import numpy as np
import anndata
import h5py
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.expanduser("~/Documents/Coding/BPTF_scDataset/scBasset/data")
MODEL_DIR  = os.path.expanduser("~/Documents/Coding/BPTF_scDataset/scBasset/models")
LOG_DIR    = os.path.expanduser("~/Documents/Coding/BPTF_scDataset/scBasset/logs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

ADATA_PATH = os.path.join(DATA_DIR, "atac_for_scbasset.h5ad")
SEQ_H5     = os.path.join(DATA_DIR, "sequences.h5")
MODEL_PATH = os.path.join(MODEL_DIR, "scbasset_model.h5")

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE  = 128
EPOCHS      = 30
LR          = 1e-3
VAL_SPLIT   = 0.1   # fraction of peaks held out for validation
SEQ_LEN     = 1344
N_FILTERS   = 288   # scBasset default
DROPOUT     = 0.2

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading AnnData...")
adata = anndata.read_h5ad(ADATA_PATH)
n_cells = adata.n_obs
n_peaks = adata.n_vars
print(f"  {n_peaks} peaks x {n_cells} cells")

print("Loading sequences...")
with h5py.File(SEQ_H5, "r") as f:
    sequences = f["sequences"][:]   # (n_peaks, 1344, 4)
print(f"  Sequences shape: {sequences.shape}")

# Accessibility matrix: peaks x cells (transpose of adata.X)
import scipy.sparse
if scipy.sparse.issparse(adata.X):
    atac_matrix = np.array(adata.X.T.todense(), dtype=np.float32)  # peaks x cells
else:
    atac_matrix = adata.X.T.astype(np.float32)

# ── Train / validation split (by peaks) ──────────────────────────────────────
peak_idx = np.arange(n_peaks)
train_idx, val_idx = train_test_split(peak_idx, test_size=VAL_SPLIT, random_state=42)
print(f"  Train peaks: {len(train_idx)}, Val peaks: {len(val_idx)}")

# ── Build scBasset model ──────────────────────────────────────────────────────
# Architecture from Yuan & Kelley 2022 (simplified):
# Conv tower → Global pooling → Dense bottleneck → n_cells output units
def build_scbasset(seq_len, n_cells, n_filters=288, dropout=0.2):
    inputs = tf.keras.Input(shape=(seq_len, 4), name="sequence")
    x = inputs

    # Stem conv block
    x = tf.keras.layers.Conv1D(n_filters, 17, padding="same", activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("gelu")(x)
    x = tf.keras.layers.MaxPooling1D(3)(x)

    # Tower of conv blocks with residual connections
    for width, dilation in [(n_filters, 1), (n_filters, 2), (n_filters, 4), (n_filters, 8)]:
        residual = x
        x = tf.keras.layers.Conv1D(width, 5, padding="same", dilation_rate=dilation, activation=None)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("gelu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Conv1D(width, 1, padding="same", activation=None)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = x + residual   # residual skip

    # Global average pooling → bottleneck (cell embedding lives here)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(32, activation="gelu", name="cell_embedding")(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    # Output: one sigmoid unit per cell
    outputs = tf.keras.layers.Dense(n_cells, activation="sigmoid", name="accessibility")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

print("\nBuilding model...")
model = build_scbasset(SEQ_LEN, n_cells, N_FILTERS, DROPOUT)
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

# ── tf.data pipeline ──────────────────────────────────────────────────────────
def make_dataset(idx, shuffle=True):
    seqs   = sequences[idx]           # (N, 1344, 4)
    labels = atac_matrix[idx]         # (N, n_cells)
    ds = tf.data.Dataset.from_tensor_slices((seqs, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=5000, seed=42)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(train_idx, shuffle=True)
val_ds   = make_dataset(val_idx, shuffle=False)

# ── Callbacks ─────────────────────────────────────────────────────────────────
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH, monitor="val_auc", mode="max",
        save_best_only=True, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=5, verbose=1,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_auc", mode="max", factor=0.5, patience=3, verbose=1
    ),
    tf.keras.callbacks.CSVLogger(os.path.join(LOG_DIR, "training_log.csv")),
]

# ── Train ─────────────────────────────────────────────────────────────────────
print(f"\nTraining for up to {EPOCHS} epochs (early stopping patience=5)...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Save training history
hist_path = os.path.join(LOG_DIR, "training_history.json")
with open(hist_path, "w") as f:
    json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)
print(f"\nTraining history saved: {hist_path}")
print(f"Best model saved: {MODEL_PATH}")
print("\nDone. Next: run 04_embeddings.py")
