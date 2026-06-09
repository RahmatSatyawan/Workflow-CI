"""
modelling.py
============
Melatih model CNN (MobileNetV2 + Custom Layers) untuk klasifikasi bunga
menggunakan MLflow Tracking UI dengan autolog.

Cara penggunaan:
    python modelling.py

Pastikan MLflow tracking server berjalan, atau biarkan MLflow menyimpan
secara lokal (default: ./mlruns).
"""

import json
from pathlib import Path
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense,
    Dropout, BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ── Konfigurasi ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
CANDIDATE_PREPROCESSING_DIRS = [
    BASE_DIR / "flowers_preprocessing",
    BASE_DIR / "preprocessing" / "flowers_preprocessing",
]
PREPROCESSING_DIR = next(
    (p.resolve() for p in CANDIDATE_PREPROCESSING_DIRS if p.exists()),
    None,
)

if PREPROCESSING_DIR is None:
    raise FileNotFoundError(
        "Tidak menemukan folder preprocessing. Periksa apakah folder flowers_preprocessing ada."
    )

TRAIN_DIR = str(PREPROCESSING_DIR / "train")
VAL_DIR = str(PREPROCESSING_DIR / "validation")
TEST_DIR = str(PREPROCESSING_DIR / "test")
METADATA_PATH = str(PREPROCESSING_DIR / "metadata.json")

IMG_HEIGHT  = 200
IMG_WIDTH   = 200
BATCH_SIZE  = 32
EPOCHS      = 20

EXPERIMENT_NAME = "flowers-classification"


# ── Muat Metadata ──────────────────────────────────────────────────────────────
def load_metadata(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File metadata tidak ditemukan: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ── Buat Data Generator ────────────────────────────────────────────────────────
def create_generators(train_dir, val_dir, test_dir,
                       img_h=IMG_HEIGHT, img_w=IMG_WIDTH,
                       batch_size=BATCH_SIZE):
    train_datagen = ImageDataGenerator(
        rescale            = 1./255,
        rotation_range     = 20,
        width_shift_range  = 0.2,
        height_shift_range = 0.2,
        shear_range        = 0.15,
        zoom_range         = 0.2,
        horizontal_flip    = True,
        fill_mode          = 'nearest'
    )
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size = (img_h, img_w),
        batch_size  = batch_size,
        class_mode  = 'categorical',
        shuffle     = True,
        seed        = 42
    )
    val_gen = val_test_datagen.flow_from_directory(
        val_dir,
        target_size = (img_h, img_w),
        batch_size  = batch_size,
        class_mode  = 'categorical',
        shuffle     = False
    )
    test_gen = val_test_datagen.flow_from_directory(
        test_dir,
        target_size = (img_h, img_w),
        batch_size  = batch_size,
        class_mode  = 'categorical',
        shuffle     = False
    )
    return train_gen, val_gen, test_gen


# ── Bangun Model ───────────────────────────────────────────────────────────────
def build_model(input_shape, num_classes):
    """
    MobileNetV2 (frozen) + Custom CNN head.
    """
    base_model = MobileNetV2(
        input_shape  = input_shape,
        include_top  = False,
        weights      = 'imagenet'
    )
    base_model.trainable = False  # Freeze base

    model = Sequential([
        base_model,
        Conv2D(64, (3, 3), activation='relu', padding='same', name='custom_conv'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ], name='MobileNetV2_Flowers')

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss      = 'categorical_crossentropy',
        metrics   = ['accuracy']
    )
    return model


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # Muat metadata
    metadata    = load_metadata(METADATA_PATH)
    class_names = metadata['class_names']
    num_classes = metadata['num_classes']

    print(f"Kelas    : {class_names}")
    print(f"N kelas  : {num_classes}")

    # Data generators
    train_gen, val_gen, test_gen = create_generators(
        TRAIN_DIR, VAL_DIR, TEST_DIR
    )

    # Bangun model
    model = build_model(
        input_shape = (IMG_HEIGHT, IMG_WIDTH, 3),
        num_classes = num_classes
    )
    model.summary()

    # MLflow setup
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.tensorflow.autolog()          # ← Autolog aktif

    callbacks = [
        EarlyStopping(
            monitor    = 'val_loss',
            patience   = 5,
            restore_best_weights = True,
            verbose    = 1
        ),
        ReduceLROnPlateau(
            monitor  = 'val_loss',
            factor   = 0.5,
            patience = 3,
            verbose  = 1
        )
    ]

    with mlflow.start_run(run_name="baseline_autolog"):
        # Simpan parameter tambahan
        mlflow.log_param("img_height",   IMG_HEIGHT)
        mlflow.log_param("img_width",    IMG_WIDTH)
        mlflow.log_param("batch_size",   BATCH_SIZE)
        mlflow.log_param("epochs",       EPOCHS)
        mlflow.log_param("num_classes",  num_classes)
        mlflow.log_param("base_model",   "MobileNetV2")
        mlflow.log_param("optimizer",    "Adam")
        mlflow.log_param("learning_rate",1e-3)

        # Latih model
        history = model.fit(
            train_gen,
            epochs            = EPOCHS,
            validation_data   = val_gen,
            callbacks         = callbacks,
            verbose           = 1
        )

        # Evaluasi pada test set
        test_loss, test_acc = model.evaluate(test_gen, verbose=0)
        print(f"\nTest Loss     : {test_loss:.4f}")
        print(f"Test Accuracy : {test_acc:.4f} ({test_acc*100:.2f}%)")

        # Log metrik test secara manual
        mlflow.log_metric("test_loss",     test_loss)
        mlflow.log_metric("test_accuracy", test_acc)

        print("\nMLflow run selesai. Buka MLflow UI:")
        print("  mlflow ui --host 0.0.0.0 --port 5000")


if __name__ == "__main__":
    main()
