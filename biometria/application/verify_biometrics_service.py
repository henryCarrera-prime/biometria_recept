# biometria/application/verify_biometrics_service.py
from typing import List, Optional
import numpy as np
from ..domain.value_objects import Thresholds, EvaluationResult

class VerifyBiometricsService:
    def __init__(
        self,
        frames: "FrameRepository",
        detector: "FaceDetector",
        glasses: "GlassesClassifier",
        antispoof: "AntiSpoof",
        liveness: "LivenessPassive",
        matcher: "SimilarityMatcher",
        thresholds: Thresholds,
    ):
        self.frames = frames
        self.detector = detector
        self.glasses = glasses
        self.antispoof = antispoof
        self.liveness = liveness
        self.matcher = matcher
        self.t = thresholds

    def execute(
        self,
        session_id: str,
        reference_url: str,
        frames_index_url: Optional[str],
        frames_urls: Optional[List[str]],
    ) -> EvaluationResult:
        urls = self.frames.list_frame_urls(frames_index_url, frames_urls)
        imgs = [self.frames.fetch_frame(u) for u in urls]
        imgs = [im for im in imgs if im is not None]
        if not imgs:
            return EvaluationResult(False, False, 0.0, 0.0, 0.0, "No frames")

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

        # Movimiento/parpadeo (regla mínima: suficientes frames válidos)
        challenge = 0.92 if valid_faces >= 10 else 0.6

        # pick best frame (usa tu métrica real de nitidez/brillo)
        best = next((f for f in analyzed if f["face_ok"]), analyzed[0])
        best_img = best["img"]

        # anti-spoof propio
        spoof_prob = self.antispoof.spoof_probability(best_img)
        antispoof_component = 1.0 - spoof_prob   # 0..1

        # luxand
        luxand = self.liveness.score(best_img)   # 0..1

        # fusión (ajusta pesos vía experimentos)
        live_score = 0.45*challenge + 0.35*antispoof_component + 0.20*luxand  # 0..1

        # similarity (descarga referencia en el repo de frames o usa otro adaptador)
        ref_img = self.frames.fetch_frame(reference_url)
        if ref_img is None:
            return EvaluationResult(False, False, live_score, 0.0, live_score*50, "Reference not found")

        similarity = self.matcher.compare(best_img, ref_img)  # 0..100

        liveness_ok = (live_score >= self.t.live) and (luxand >= self.t.luxand)
        match_ok = (similarity >= self.t.similarity)

        evaluation_pct = 0.5*(live_score*100) + 0.5*similarity
        passed = liveness_ok and match_ok and (evaluation_pct >= 95.0)
        msg = "OK" if passed else ("Liveness insuficiente" if not liveness_ok else ("Baja similitud" if not match_ok else "Evaluación < 95%"))
        return EvaluationResult(liveness_ok, match_ok, live_score, similarity, evaluation_pct, msg)
