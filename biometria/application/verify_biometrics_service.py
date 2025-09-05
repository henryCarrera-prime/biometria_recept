# biometria/application/verify_biometrics_service.py
from typing import List
from ..domain.value_objects import Thresholds, EvaluationResult
from ..domain.interfaces import (
    ReferenceImageRepository, FramesRepository, FaceDetector,
    GlassesClassifier, AntiSpoof, LivenessPassive, SimilarityMatcher
)

class VerifyBiometricsService:
    def __init__(
        self,
        reference_repo: ReferenceImageRepository,
        frames_repo: FramesRepository,
        detector: FaceDetector,
        glasses: GlassesClassifier,
        antispoof: AntiSpoof,
        liveness: LivenessPassive,
        matcher: SimilarityMatcher,
        thresholds: Thresholds,
    ):
        self.reference_repo = reference_repo
        self.frames_repo = frames_repo
        self.detector = detector
        self.glasses = glasses
        self.antispoof = antispoof
        self.liveness = liveness
        self.matcher = matcher
        self.t = thresholds

    def execute(self, uuid_proceso: str, reference_dir: str, frames_dir: str) -> EvaluationResult:
        # 1) referencia (UNA imagen dentro de la carpeta reference_dir)
        ref_img = self.reference_repo.fetch_reference(reference_dir)
        if ref_img is None:
            return EvaluationResult(False, False, 0.0, 0.0, 0.0, "Reference not found")

        # 2) frames (N imágenes dentro de frames_dir)
        frame_paths = self.frames_repo.list_frames(frames_dir)
        if not frame_paths:
            return EvaluationResult(False, False, 0.0, 0.0, 0.0, "No frames found")

        imgs = [self.frames_repo.fetch_frame(p) for p in frame_paths]
        imgs = [im for im in imgs if im is not None]
        if not imgs:
            return EvaluationResult(False, False, 0.0, 0.0, 0.0, "Frames unreadable")

        analyzed = []
        valid_faces = 0
        glasses_all_valid = True

        for img in imgs:
            det = self.detector.detect(img)
            face_ok = det is not None
            wears = self.glasses.has_glasses(img, det["landmarks"]) if face_ok else False
            if face_ok:
                valid_faces += 1
                if not wears:
                    glasses_all_valid = False
            analyzed.append({"img": img, "face_ok": face_ok, "glasses": wears})

        if valid_faces > 0 and glasses_all_valid:
            return EvaluationResult(False, False, 0.0, 0.0, 0.0, "Glasses in all valid frames")

        # Reto mínimo (sustituye con EAR/MAR + yaw later)
        challenge = 0.92 if valid_faces >= 10 else 0.6

        # Best frame: el primero con rostro (puedes mejorarlo con nitidez/bright)
        best = next((f for f in analyzed if f["face_ok"]), analyzed[0])
        best_img = best["img"]

        # Anti-spoof + Luxand
        spoof_prob = self.antispoof.spoof_probability(best_img)    # 0..1 (0=real)
        antispoof_component = 1.0 - spoof_prob                     # 0..1
        luxand = self.liveness.score(best_img)                     # 0..1

        # Liveness final
        live_score = 0.45*challenge + 0.35*antispoof_component + 0.20*luxand  # 0..1

        # Similaridad (AWS)
        similarity = self.matcher.compare(best_img, ref_img)       # 0..100

        liveness_ok = (live_score >= self.t.live) and (luxand >= self.t.luxand)
        match_ok    = (similarity >= self.t.similarity)

        evaluation_pct = 0.5*(live_score*100) + 0.5*similarity
        passed = liveness_ok and match_ok and (evaluation_pct >= 95.0)
        msg = "OK" if passed else ("Liveness insuficiente" if not liveness_ok else ("Baja similitud" if not match_ok else "Evaluación < 95%"))
        return EvaluationResult(liveness_ok, match_ok, live_score, similarity, evaluation_pct, msg)
