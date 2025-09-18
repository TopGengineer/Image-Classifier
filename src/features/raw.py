import numpy as np
from PIL import Image
from sklearn.base import BaseEstimator, TransformerMixin

class RawPixelsFeatures(BaseEstimator, TransformerMixin):
    """Grayscale -> resize -> flatten (0..1)."""
    def __init__(self, side: int = 64):
        self.side = side

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        vecs = []
        for img in X:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.asarray(img, dtype=np.uint8))
            g = img.convert("L").resize((self.side, self.side))
            arr = np.asarray(g, dtype=np.float32) / 255.0
            vecs.append(arr.ravel())
        return np.vstack(vecs)
