# biometria/domain/interfaces.py
from __future__ import annotations
from typing import Protocol, List, Optional, Tuple
import numpy as np

# ---- Repositorios de I/O (puertos) ----

class ReferenceImageRepository(Protocol):
    """Obtiene UNA sola imagen de referencia desde una URI (s3://, file://, https://...)."""
    def fetch_reference(self, uri: str) -> Optional[np.ndarray]:
        ...

class FramesRepository(Protocol):
    """Lista y descarga N frames desde una carpeta (s3://bucket/prefix/, file://..., o índice JSON)."""
    def list_frames(self, uri: str) -> List[str]:
        ...
    def fetch_frame(self, uri: str) -> Optional[np.ndarray]:
        ...

# (Opcional) Mantén este para retro-compatibilidad si ya lo usas en algún lado:
class FrameRepository(Protocol):
    """[DEPRECADO] Unifica listado y descarga. Prefiere FramesRepository + ReferenceImageRepository."""
    def list_frame_urls(self, index_url: str | None, urls: List[str] | None) -> List[str]:
        ...
    def fetch_frame(self, url: str) -> Optional[np.ndarray]:
        ...

# ---- ML / visión (puertos) ----

class FaceDetector(Protocol):
    def detect(self, img_bgr: np.ndarray) -> Optional[dict]:
        """
        Debe devolver:
        {
          "bbox": (x1, y1, x2, y2),
          "landmarks": List[Tuple[float, float]]  # opcional
        }
        o None si no hay rostro.
        """
        ...

class GlassesClassifier(Protocol):
    def has_glasses(self, img_bgr: np.ndarray, landmarks=None) -> bool:
        ...

class AntiSpoof(Protocol):
    def spoof_probability(self, face_bgr: np.ndarray) -> float:
        """0.0 = real, 1.0 = spoof."""
        ...

class LivenessPassive(Protocol):
    def score(self, face_bgr: np.ndarray) -> float:
        """0..1."""
        ...

class SimilarityMatcher(Protocol):
    def compare(self, probe_bgr: np.ndarray, reference_bgr: np.ndarray) -> float:
        """Devuelve similitud 0..100 (compat con Rekognition CompareFaces)."""
        ...
