# biometria/infrastructure/config.py
import os
from ..domain.value_objects import Thresholds
from ..application.verify_biometrics_service import VerifyBiometricsService
from .storage.http_frame_repository import HttpFrameRepository
from .similarity.rekognition_adapter import RekognitionMatcher
from .liveness.luxand_client import LuxandClient

def build_verify_service():
    frames = HttpFrameRepository()
    # TODO: reemplazar por implementaciones reales
    detector = DummyFaceDetector()
    glasses = DummyGlasses()
    antispoof = DummyAntiSpoof()
    liveness = LuxandClient(token=os.getenv("LUXAND_TOKEN",""))
    matcher  = RekognitionMatcher(region=os.getenv("AWS_REGION","us-east-1"),
                                  similarity_th=float(os.getenv("SIMILARITY_TH","95")))
    thresholds = Thresholds(
        similarity=float(os.getenv("SIMILARITY_TH","95")),
        live=float(os.getenv("LIVE_TH","0.90")),
        luxand=float(os.getenv("LUXAND_LIVENESS_TH","0.85")),
    )
    return VerifyBiometricsService(frames, detector, glasses, antispoof, liveness, matcher, thresholds)

# Dummies (mientras cableas los reales)
class DummyFaceDetector:  # implements FaceDetector
    def detect(self, img_bgr): return {"bbox": (0,0,10,10), "landmarks": None}
class DummyGlasses:       # implements GlassesClassifier
    def has_glasses(self, img_bgr, landmarks=None): return False
class DummyAntiSpoof:     # implements AntiSpoof
    def spoof_probability(self, img_bgr): return 0.2
