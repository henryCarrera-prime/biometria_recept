# biometria/domain/interfaces.py
from typing import Protocol, List, Optional
import numpy as np

class FrameRepository(Protocol):
    def list_frame_urls(self, index_url: str | None, urls: List[str] | None) -> List[str]:
        ...
    def fetch_frame(self, url: str) -> Optional[np.ndarray]: ...

class FaceDetector(Protocol):
    def detect(self, img_bgr: np.ndarray) -> Optional[dict]: ...
    # dict: { "bbox": (x1,y1,x2,y2), "landmarks": [(x,y), ...] }

class GlassesClassifier(Protocol):
    def has_glasses(self, img_bgr: np.ndarray, landmarks=None) -> bool: ...

class AntiSpoof(Protocol):
    def spoof_probability(self, face_bgr: np.ndarray) -> float: ...

class LivenessPassive(Protocol):
    def score(self, face_bgr: np.ndarray) -> float: ...

class SimilarityMatcher(Protocol):
    # devuelve similitud 0..100
    def compare(self, probe_bgr: np.ndarray, reference_bgr: np.ndarray) -> float: ...
