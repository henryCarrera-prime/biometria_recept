# biometria/infrastructure/config.py
import os
from ..domain.value_objects import Thresholds
from ..application.verify_biometrics_service import VerifyBiometricsService

# Repos locales (carpetas Windows)
from .storage.local_repositories import LocalReferenceRepository, LocalFramesRepository
from .storage.s3_repositories import SmartReferenceRepository, SmartFramesRepository
# Adaptadores
from .similarity.rekognition_adapter import RekognitionMatcher
from .liveness.luxand_client import LuxandClient
from .detection.rekognation_face_detector import RekognitionFaceDetector
# Dummies (mientras cableas modelos reales)

class DummyGlasses:       # implements GlassesClassifier
    def has_glasses(self, img_bgr, landmarks=None): return False

class DummyAntiSpoof:     # implements AntiSpoof
    def spoof_probability(self, img_bgr): return 0.20  # 20% spoof

class NoopLiveness:
    def score(self, face_bgr): return 0.0

def get_thresholds() -> Thresholds:
    return Thresholds(
        similarity=float(os.getenv("SIMILARITY_TH", "95")),
        live=float(os.getenv("LIVE_TH", "0.90")),
        luxand=float(os.getenv("LUXAND_LIVENESS_TH", "0.85")),
    )
def _common_detector_and_matcher(thresholds: Thresholds):
    detector = RekognitionFaceDetector(region=os.getenv("AWS_REGION", "us-east-1"))
    matcher = RekognitionMatcher(
        region=os.getenv("AWS_REGION", "us-east-1"),
        similarity_th=thresholds.similarity
    )
    return detector, matcher
def build_verify_service_for_local_dirs() -> VerifyBiometricsService:
    thresholds = get_thresholds()
    detector, matcher = _common_detector_and_matcher(thresholds)
    return VerifyBiometricsService(
        reference_repo=LocalReferenceRepository(),
        frames_repo=LocalFramesRepository(),
        detector=detector,
        glasses=DummyGlasses(),         # TODO: clasificador gafas
        antispoof=DummyAntiSpoof(),     # TODO: anti-spoof real
        liveness=LuxandClient(token=os.getenv("LUXAND_TOKEN","")),
        matcher=RekognitionMatcher(
            region=os.getenv("AWS_REGION","us-east-1"),
            similarity_th=thresholds.similarity
        ),
        thresholds=thresholds
    )

def build_verify_service_auto() -> VerifyBiometricsService:
    thresholds = get_thresholds()
    detector, matcher = _common_detector_and_matcher(thresholds)
    return VerifyBiometricsService(
        reference_repo=SmartReferenceRepository(),
        frames_repo=SmartFramesRepository(),
        detector=detector,
        glasses=DummyGlasses(),
        antispoof=DummyAntiSpoof(),
        liveness=LuxandClient(token=os.getenv("LUXAND_TOKEN", "")),
        matcher=matcher,
        thresholds=thresholds
    )