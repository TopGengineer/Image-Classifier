import numpy as np
from PIL import Image
from scipy.fftpack import dctn
from sklearn.base import BaseEstimator, TransformerMixin

class TextureFeaturesCompact(BaseEstimator, TransformerMixin):
    """Grayscale -> resize -> 2D DCT -> keep low-freq kxk -> flatten."""
    def __init__(self, side: int = 64, k: int = 16):
        self.side, self.k = side, k

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = []
        for img in X:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.asarray(img, dtype=np.uint8))
            arr = np.asarray(img.convert("L").resize((self.side, self.side)), dtype=float) / 255.0
            d = dctn(arr, type=2, norm="ortho")
            kk = min(self.k, d.shape[0])
            out.append(d[:kk, :kk].ravel())
        return np.vstack(out)
