import os, pathlib, re
from typing import List, Optional
import cv2, numpy as np

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

def normalize_win_dir(p: str) -> str:
    # Acepta "C:stalin/carpeta" y lo corrige a "C:\stalin\carpeta\"
    p = p.strip().replace("/", "\\")
    m = re.match(r"^[A-Za-z]:(?![\\/])", p)  # ej. "C:" seguido de no slash
    if m:
        p = p[:2] + "\\" + p[2:]
    p = os.path.normpath(p)
    if not p.endswith(os.sep):
        p = p + os.sep
    return p

def list_images_in_dir(dir_path: str) -> List[str]:
    base = pathlib.Path(normalize_win_dir(dir_path))
    if not base.exists() or not base.is_dir():
        return []
    files = []
    for ext in IMG_EXT:
        files.extend(str(p) for p in base.glob(f"*{ext}"))
    return sorted(files)

def load_local_image(path_str: str) -> Optional[np.ndarray]:
    path = os.path.normpath(path_str)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            data = f.read()
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def variance_of_laplacian(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def pick_reference_single(dir_path: str) -> Optional[str]:
    imgs = list_images_in_dir(dir_path)
    if not imgs:
        return None
    if len(imgs) == 1:
        return imgs[0]
    # Si hubiera más de una por error, intenta "reference.*" o la más nítida
    for p in imgs:
        name = os.path.basename(p).lower()
        if name.startswith("reference."):
            return p
    best, best_sharp = None, -1.0
    for p in imgs[:50]:
        im = load_local_image(p)
        if im is None:
            continue
        sh = variance_of_laplacian(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
        if sh > best_sharp:
            best, best_sharp = p, sh
    return best or imgs[0]

# Implementaciones de puertos
class LocalReferenceRepository:
    def fetch_reference(self, uri: str) -> Optional[np.ndarray]:
        # uri es carpeta con UNA imagen
        ref_path = pick_reference_single(uri)
        if not ref_path:
            return None
        return load_local_image(ref_path)

class LocalFramesRepository:
    def list_frames(self, uri: str) -> List[str]:
        return list_images_in_dir(uri)

    def fetch_frame(self, uri: str) -> Optional[np.ndarray]:
        return load_local_image(uri)
