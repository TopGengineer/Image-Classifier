from typing import Tuple, List
import numpy as np
from PIL import Image

try:
    import tensorflow as tf
    from tensorflow import keras
    HAVE_TF = True
except Exception:
    HAVE_TF = False

from sklearn.preprocessing import LabelEncoder

def prep_arrays(imgs: List[Image.Image], labels: np.ndarray, side: int):
    le = LabelEncoder()
    yint = le.fit_transform(labels)
    X = np.zeros((len(imgs), side, side, 1), dtype="float32")
    for i, im in enumerate(imgs):
        g = np.asarray(im.convert("L").resize((side, side)), dtype=np.float32) / 255.0
        X[i, :, :, 0] = g
    return X, yint, le

def build_small_cnn(input_shape: Tuple[int,int,int], n_classes: int, lr: float = 1e-3):
    if not HAVE_TF:
        raise RuntimeError("TensorFlow not installed.")
    model = keras.Sequential([
        keras.layers.Conv2D(16, 3, activation="relu", padding="same", input_shape=input_shape),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.25),
        keras.layers.Dense(n_classes, activation="softmax")
    ])
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
