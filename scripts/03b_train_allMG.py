"""
03b_train_allMG.py
Retrain scBasset on the full mammary gland dataset (all cell types, WT + KO).
Mirrors 03_train_scbasset.py with paths updated for allMG data.
Expected improvement over epithelial-only model: val AUC ~0.78-0.85.
"""

import os, json
import numpy as np
import anndata
import h5py
import scipy.sparse
import tensorflow as tf
from sklearn.model_selection import train_test_split

DATA_DIR  = os.path.expanduser("~/Documents/Coding/BPTF_scDataset/scBasset/data/allMG")
MODEL_DIR = os.path.expanduser("~/Documents/Coding/BPTF_scDataset/scBasset/models")
LOG_DIR   = os.path.expanduser("~/Documents/Coding/BPTF_scDataset/scBasset/logs")
os.makedirs(MODEL_DIR, exist_ok=True)

ADATA_PATH  = os.path.join(DATA_DIR, "allMG_for_scbasset.h5ad")
SEQ_H5      = os.path.join(DATA_DIR, "allMG_sequences.h5")
MODEL_PATH  = os.path.join(MODEL_DIR, "scbasset_allMG_model.h5")

BATCH_SIZE  = 128
EPOCHS      = 30
LR          = 1e-3
VAL_SPLIT   = 0.1
SEQ_LEN     = 1344
N_FILTERS   = 288
DROPOUT     = 0.2

print("Loading AnnData...")
adata = anndata.read_h5ad(ADATA_PATH)
n_cells, n_peaks = adata.n_obs, adata.n_vars
print(f"  {n_peaks} peaks x {n_cells} cells")
print(f"  Cell types: {adata.obs['cell_type'].value_counts().to_dict()}")
print(f"  Conditions: {adata.obs['condition'].value_counts().to_dict()}")

print("Loading sequences...")
with h5py.File(SEQ_H5, "r") as f:
    sequences = f["sequences"][:]
print(f"  Sequences: {sequences.shape}")

atac_matrix = (np.array(adata.X.T.todense(), dtype=np.float32)
               if scipy.sparse.issparse(adata.X)
               else adata.X.T.astype(np.float32))

train_idx, val_idx = train_test_split(
    np.arange(n_peaks), test_size=VAL_SPLIT, random_state=42
)
print(f"  Train: {len(train_idx)} peaks | Val: {len(val_idx)} peaks")

# ── Model ─────────────────────────────────────────────────────────────────────
def build_scbasset(seq_len, n_cells, n_filters=288, dropout=0.2):
    inputs = tf.keras.Input(shape=(seq_len, 4), name="sequence")
    x = inputs
    x = tf.keras.layers.Conv1D(n_filters, 17, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("gelu")(x)
    x = tf.keras.layers.MaxPooling1D(3)(x)
    for dilation in [1, 2, 4, 8]:
        res = x
        x = tf.keras.layers.Conv1D(n_filters, 5, padding="same", dilation_rate=dilation)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("gelu")(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Conv1D(n_filters, 1, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = x + res
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(32, activation="gelu", name="cell_embedding")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(n_cells, activation="sigmoid", name="accessibility")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

print("\nBuilding model...")
model = build_scbasset(SEQ_LEN, n_cells, N_FILTERS, DROPOUT)
model.summary()
model.compile(
    optimizer=tf.keras.optimizers.Adam(LR),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

def make_dataset(idx, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices(
        (sequences[idx], atac_matrix[idx])
    )
    if shuffle:
        ds = ds.shuffle(5000, seed=42)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(train_idx, shuffle=True)
val_ds   = make_dataset(val_idx,   shuffle=False)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH, monitor="val_auc", mode="max", save_best_only=True, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=5, verbose=1, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_auc", mode="max", factor=0.5, patience=3, verbose=1
    ),
    tf.keras.callbacks.CSVLogger(os.path.join(LOG_DIR, "training_allMG_log.csv")),
]

print(f"\nTraining on full mammary gland dataset ({n_cells} cells, {n_peaks} peaks)...")
history = model.fit(
    train_ds, validation_data=val_ds,
    epochs=EPOCHS, callbacks=callbacks, verbose=1
)

hist_path = os.path.join(LOG_DIR, "training_allMG_history.json")
with open(hist_path, "w") as f:
    json.dump({k: [float(v) for v in vals]
               for k, vals in history.history.items()}, f, indent=2)

print(f"\nBest model saved: {MODEL_PATH}")
print(f"Training history: {hist_path}")
print("Done. Next: run 04_embeddings.py (update MODEL_PATH and ADATA_PATH to allMG versions)")
