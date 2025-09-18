import io, os, zipfile
from typing import List, Dict, Optional, Tuple
import numpy as np
from PIL import Image

IMG_EXT = {".png",".jpg",".jpeg",".bmp",".tif",".tiff",".dcm",".dicom",
           ".PNG",".JPG",".JPEG",".BMP",".TIF",".TIFF",".DCM",".DICOM"}

try:
    import pydicom
    HAVE_PYDICOM = True
except Exception:
    HAVE_PYDICOM = False

def _is_img(path: str) -> bool:
    return any(path.endswith(ext) for ext in IMG_EXT)

def _clean_parts(parts: List[str]) -> List[str]:
    return [p for p in parts if p and not p.startswith("__MACOSX") and p != ".DS_Store"]

def _decode_image(name: str, raw: bytes):
    ext = os.path.splitext(name)[1].lower()
    if ext in (".dcm",".dicom"):
        if not HAVE_PYDICOM: return None
        try:
            ds = pydicom.dcmread(io.BytesIO(raw))
            arr = ds.pixel_array.astype(np.float32)
            if arr.ndim == 3:
                arr = arr[arr.shape[0] // 2]
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            arr = (arr * slope + intercept)
            arr = arr - np.min(arr)
            if arr.max() > 0: arr = arr / arr.max()
            return Image.fromarray((arr * 255).astype(np.uint8)).convert("RGB")
        except Exception:
            return None
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return None

def _zip_img_names(zf: zipfile.ZipFile) -> List[str]:
    return [n for n in zf.namelist() if not n.endswith("/") and _is_img(n)]

def _suggest_roots(names: List[str], min_per_class=2) -> Dict[str, Dict[str,int]]:
    cands: Dict[str, Dict[str,int]] = {}
    for n in names:
        parts = _clean_parts(n.split("/"))
        for i in range(len(parts)-1):
            root = "/".join(parts[:i+1])
            if not root or i+1 >= len(parts)-1:
                continue
            lab = parts[i+1]
            cands.setdefault(root, {}).setdefault(lab, 0)
            cands[root][lab] += 1
    return {r:d for r,d in cands.items() if sum(v>=min_per_class for v in d.values()) >= 2}

def load_from_zip(zip_bytes: bytes, chosen_root: Optional[str], limit_per_class: Optional[int]=None
                 ) -> Tuple[List[Image.Image], np.ndarray, List[str], Dict]:
    X, y = [], []
    report = {"read_ok":0, "skipped":0, "class_counts":{}}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        names = _zip_img_names(z)
        if not names:
            raise ValueError("No image files found in ZIP.")
        roots = _suggest_roots(names, min_per_class=2)
        if not roots:
            raise ValueError("Could not detect `<ROOT>/<class>/*` structure.")
        if chosen_root is None:
            chosen_root = max(roots.keys(), key=lambda r: sum(roots[r].values()))
        depth = len([p for p in chosen_root.split("/") if p])

        loaded_per: Dict[str,int] = {}
        for n in names:
            parts = _clean_parts(n.split("/"))
            if len(parts) <= depth or "/".join(parts[:depth]) != chosen_root:
                continue
            label = parts[depth]
            if limit_per_class and loaded_per.get(label,0) >= limit_per_class:
                continue
            try:
                with z.open(n) as f:
                    raw = f.read()
                img = _decode_image(n, raw)
                if img is None:
                    report["skipped"] += 1
                    continue
                X.append(img); y.append(label)
                loaded_per[label] = loaded_per.get(label,0) + 1
                report["read_ok"] += 1
                report["class_counts"][label] = report["class_counts"].get(label, 0) + 1
            except Exception:
                report["skipped"] += 1

    classes = sorted(set(y))
    if len(classes) < 2:
        raise ValueError("Need at least 2 classes under the chosen root.")
    import numpy as np  # local import for type clarity
    return X, np.array(y), classes, report

def list_roots_in_zip(zip_bytes: bytes) -> Dict[str, Dict[str,int]]:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        names = _zip_img_names(z)
        return _suggest_roots(names, min_per_class=2)
