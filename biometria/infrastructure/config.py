# biometria/infrastructure/config.py
import os
from ..domain.value_objects import Thresholds
from ..application.verify_biometrics_service import VerifyBiometricsService

# Repos locales (carpetas Windows)
from .storage.local_repositories import LocalReferenceRepository, LocalFramesRepository

# Adaptadores
from .similarity.rekognition_adapter import RekognitionMatcher
from .liveness.luxand_client import LuxandClient
import cv2
# Dummies (mientras cableas modelos reales)
class DummyFaceDetector:  # implements FaceDetector
    def detect(self, img_bgr):
       
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        sh = cv2.Laplacian(gray, cv2.CV_64F).var()
        return {"bbox": (0,0,10,10), "landmarks": None} if sh >= 30 else None

class DummyGlasses:       # implements GlassesClassifier
    def has_glasses(self, img_bgr, landmarks=None): return False

class DummyAntiSpoof:     # implements AntiSpoof
    def spoof_probability(self, img_bgr): return 0.20  # 20% spoof

class NoopLiveness:
    def score(self, face_bgr): return 0.0

def build_verify_service_for_local_dirs() -> VerifyBiometricsService:
    thresholds = Thresholds(
        similarity=float(os.getenv("SIMILARITY_TH","95")),
        live=float(os.getenv("LIVE_TH","0.90")),
        luxand=float(os.getenv("LUXAND_LIVENESS_TH","0.85")),
    )

    return VerifyBiometricsService(
        reference_repo=LocalReferenceRepository(),
        frames_repo=LocalFramesRepository(),
        detector=DummyFaceDetector(),   # TODO: detector real (RetinaFace/SCRFD)
        glasses=DummyGlasses(),         # TODO: clasificador gafas
        antispoof=DummyAntiSpoof(),     # TODO: anti-spoof real
        liveness=LuxandClient(token=os.getenv("LUXAND_TOKEN","")),
        matcher=RekognitionMatcher(
            region=os.getenv("AWS_REGION","us-east-1"),
            similarity_th=thresholds.similarity
        ),
        thresholds=thresholds
    )
