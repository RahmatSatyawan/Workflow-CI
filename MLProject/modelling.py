"""
modelling.py
============
Melatih model CNN (MobileNetV2 + Custom Layers) untuk klasifikasi bunga
menggunakan MLflow Tracking UI dengan autolog.

Cara penggunaan:
    python modelling.py --epochs 10 --batch_size 32 --img_size 200 --data_dir ../../preprocessing/flowers_preprocessing
"""

import argparse
import json
from pathlib import Path

import tensorflow as tf
import mlflow
import mlflow.tensorflow

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense,
    Dropout, BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

BASE_DIR = Path(__file__).resolve().parent
EXPERIMENT_NAME = "flowers-classification"


def parse_args():
    parser = argparse.ArgumentParser(description="Train flowers classifier with MLflow")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=200)
    parser.add_argument("--data_dir", type=str, default="flowers_preprocessing")
    parser.add_argument('--alpha', type=float, default=0.01, help="Alpha parameter for MLflow")
    parser.add_argument('--l1_ratio', type=float, default=0.5, help="L1 ratio parameter for MLflow")
    return parser.parse_args()


def resolve_preprocessing_dir(data_dir: str) -> Path:
    raw_path = Path(data_dir)
    candidates = []

    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend([
            BASE_DIR / raw_path,
            BASE_DIR / "preprocessing" / raw_path,
            BASE_DIR.parent.parent / raw_path,
            BASE_DIR.parent.parent / "preprocessing" / raw_path,
            BASE_DIR.parent / raw_path,
        ])

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Tidak menemukan folder preprocessing untuk data_dir={data_dir}. "
        f"Dicoba: {', '.join(str(c) for c in candidates)}"
    )


def load_metadata(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File metadata tidak ditemukan: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_generators(train_dir, val_dir, test_dir, img_h, img_w, batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_h, img_w),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )
    val_gen = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=(img_h, img_w),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    test_gen = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_h, img_w),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    return train_gen, val_gen, test_gen


def build_model(input_shape, num_classes, learning_rate):
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

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
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def main():
    args = parse_args()
    preprocessing_dir = resolve_preprocessing_dir(args.data_dir)

    train_dir = preprocessing_dir / 'train'
    val_dir = preprocessing_dir / 'validation'
    test_dir = preprocessing_dir / 'test'
    metadata_path = preprocessing_dir / 'metadata.json'

    metadata = load_metadata(metadata_path)
    class_names = metadata['class_names']
    num_classes = metadata['num_classes']

    print(f"Kelas    : {class_names}")
    print(f"N kelas  : {num_classes}")

    train_gen, val_gen, test_gen = create_generators(
        str(train_dir),
        str(val_dir),
        str(test_dir),
        img_h=args.img_size,
        img_w=args.img_size,
        batch_size=args.batch_size
    )

    model = build_model(
        input_shape=(args.img_size, args.img_size, 3),
        num_classes=num_classes,
        learning_rate=args.learning_rate
    )
    model.summary()

    # mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.tensorflow.autolog()

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1
        )
    ]

    with mlflow.start_run(run_name="baseline_autolog"):
        mlflow.log_param("img_height", args.img_size)
        mlflow.log_param("img_width", args.img_size)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("base_model", "MobileNetV2")
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("learning_rate", args.learning_rate)

        model.fit(
            train_gen,
            epochs=args.epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )

        test_loss, test_acc = model.evaluate(test_gen, verbose=0)
        print(f"\nTest Loss     : {test_loss:.4f}")
        print(f"Test Accuracy : {test_acc:.4f} ({test_acc*100:.2f}%)")

        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)

        print("\nMLflow run selesai. Buka MLflow UI:")
        print("  mlflow ui --host 0.0.0.0 --port 5000")


if __name__ == "__main__":
    main()
