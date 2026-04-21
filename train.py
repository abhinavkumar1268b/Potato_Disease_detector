import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib
matplotlib.use("Agg")           # Non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt

# ─── Configuration ────────────────────────────────────────────────────────────

# Path to the dataset root folder
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data set")

# Image parameters (must match original model setup)
IMAGE_SIZE   = (256, 256)
BATCH_SIZE   = 32
CHANNELS     = 3
EPOCHS       = 50

# Train / Validation / Test split ratios
TRAIN_SPLIT = 0.8
VAL_SPLIT   = 0.1
# Test = remaining 0.1

# Model output paths
MODEL_SAVE_PATH     = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "potato_disease_model_v2.keras")
APP_MODEL_DEST      = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "app", "potato_disease_model_v2.keras")

# ─── Step 1: Load Dataset ─────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  POTATO DISEASE CLASSIFIER — TRAINING (4 Classes)")
print("=" * 60)

print(f"\n[1] Loading dataset from: {DATASET_DIR}")

# Load the full dataset (all images, shuffled)
full_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR,
    shuffle=True,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    seed=42
)

# Print discovered class names
class_names = full_dataset.class_names
num_classes = len(class_names)
print(f"    Classes found ({num_classes}): {class_names}")

# Count total batches for splitting
total_batches = tf.data.experimental.cardinality(full_dataset).numpy()
print(f"    Total batches: {total_batches}")

# ─── Step 2: Split into Train / Val / Test ────────────────────────────────────

print("\n[2] Splitting dataset ...")

train_size = int(TRAIN_SPLIT * total_batches)   # 80%
val_size   = int(VAL_SPLIT   * total_batches)   # 10%
# test_size  = remainder                         # 10%

train_ds = full_dataset.take(train_size)
remaining = full_dataset.skip(train_size)
val_ds   = remaining.take(val_size)
test_ds  = remaining.skip(val_size)

print(f"    Train batches : {tf.data.experimental.cardinality(train_ds).numpy()}")
print(f"    Val batches   : {tf.data.experimental.cardinality(val_ds).numpy()}")
print(f"    Test batches  : {tf.data.experimental.cardinality(test_ds).numpy()}")

# Optimise pipeline performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds  = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ─── Step 3: Build the Model ──────────────────────────────────────────────────

print("\n[3] Building model ...")

# Data augmentation — helps model generalise to real-world uploads
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.1),
], name="data_augmentation")

# Normalisation — scale pixel values from [0,255] to [0,1]
# (Baked into the model so no preprocessing needed at inference time)
rescaling = layers.Rescaling(1.0 / 255, name="rescaling")

# CNN architecture — same style as the original notebook, extended for 4 classes
model = models.Sequential([
    # Input shape
    layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], CHANNELS)),

    # Preprocessing (inside model — portable to Streamlit with no extra steps)
    data_augmentation,
    rescaling,

    # Block 1
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    # Block 2
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    # Block 3
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    # Block 4
    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    # Block 5
    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    # Classifier head
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),                        # Regularisation
    layers.Dense(num_classes, activation="softmax"),  # 4 output nodes
], name="potato_disease_cnn_v2")

model.summary()

# ─── Step 4: Compile ──────────────────────────────────────────────────────────

print("\n[4] Compiling model ...")

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

# ─── Step 5: Callbacks ────────────────────────────────────────────────────────

callbacks = [
    # Stop early if val_accuracy doesn't improve for 5 epochs
    EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    # Save the best model during training
    ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
]

# ─── Step 6: Train ────────────────────────────────────────────────────────────

print(f"\n[5] Training for up to {EPOCHS} epochs ...")
print(f"    (EarlyStopping: patience=5 on val_accuracy)\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ─── Step 7: Evaluate on Test Set ─────────────────────────────────────────────

print("\n[6] Evaluating on test set ...")
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"\n    Test Loss     : {test_loss:.4f}")
print(f"    Test Accuracy : {test_acc * 100:.2f}%")

# ─── Step 8: Class-wise Prediction Report ─────────────────────────────────────

print("\n[7] Generating class-wise prediction report ...")

all_labels      = []
all_predictions = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    all_labels.extend(labels.numpy())
    all_predictions.extend(np.argmax(preds, axis=1))

all_labels      = np.array(all_labels)
all_predictions = np.array(all_predictions)

# Print class-wise accuracy manually (no sklearn required)
print("\n  Class-wise Results:")
print(f"  {'Class':<30} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
print("  " + "-" * 60)

for i, cls in enumerate(class_names):
    mask    = (all_labels == i)
    total   = int(mask.sum())
    correct = int((all_predictions[mask] == i).sum())
    acc     = (correct / total * 100) if total > 0 else 0.0
    print(f"  {cls:<30} {correct:>8} {total:>8} {acc:>9.1f}%")

print()

# ─── Step 9: Save Training Plots ──────────────────────────────────────────────

print("[8] Saving training plots ...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
ax1.plot(history.history["accuracy"],     label="Train Accuracy",  color="#2196F3")
ax1.plot(history.history["val_accuracy"], label="Val Accuracy",    color="#4CAF50")
ax1.set_title("Model Accuracy", fontsize=14)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Loss plot
ax2.plot(history.history["loss"],     label="Train Loss", color="#F44336")
ax2.plot(history.history["val_loss"], label="Val Loss",   color="#FF9800")
ax2.set_title("Model Loss", fontsize=14)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "training_results.png")
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"    Saved plot → {plot_path}")

# ─── Step 10: Copy Model to app/ ──────────────────────────────────────────────

print("\n[9] Copying model to app/ folder for Streamlit ...")
try:
    shutil.copy2(MODEL_SAVE_PATH, APP_MODEL_DEST)
    print(f"    Copied → {APP_MODEL_DEST}")
except Exception as e:
    print(f"    [WARNING] Could not copy model to app/: {e}")
    print(f"    Please manually copy '{MODEL_SAVE_PATH}' to 'app/'")

# ─── Done ─────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  TRAINING COMPLETE")
print(f"  Model saved  : {MODEL_SAVE_PATH}")
print(f"  Test accuracy: {test_acc * 100:.2f}%")
print(f"  Classes      : {class_names}")
print("=" * 60 + "\n")
