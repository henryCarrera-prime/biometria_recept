# biometria/application/verify_biometrics_service.py
from typing import List
from ..domain.value_objects import Thresholds, EvaluationResult
from ..domain.interfaces import (
    ReferenceImageRepository, FramesRepository, FaceDetector,
    GlassesClassifier, AntiSpoof, LivenessPassive, SimilarityMatcher
)
import cv2, numpy as np

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
        """
        Verifica biometría facial en un proceso de verificación.

        Parámetros:
        - uuid_proceso: Identificador único del proceso (no usado aquí, pero útil para logs).
        - reference_dir: Ruta al directorio que contiene la imagen de referencia.
        - frames_dir: Ruta al directorio que contiene las imágenes capturadas (frames).

        Retorna:
        - EvaluationResult con los resultados de la verificación.
        """

    # Helpers locales
        def sharpness(img_bgr) -> float:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())

        def brightness(img_bgr) -> float:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            return float(np.mean(gray))

    # 1) referencia (UNA imagen en la carpeta reference_dir)
        ref_img = self.reference_repo.fetch_reference(reference_dir)
        if ref_img is None:
            return EvaluationResult(False, False, 0.0, 0.0, 0.0, "Reference not found")

    # 2) frames (N imágenes en la carpeta frames_dir)
        frame_paths = self.frames_repo.list_frames(frames_dir)
        if not frame_paths:
            return EvaluationResult(False, False, 0.0, 0.0, 0.0, "No frames found")

        imgs = [self.frames_repo.fetch_frame(p) for p in frame_paths]
        imgs = [im for im in imgs if im is not None]
        if not imgs:
            return EvaluationResult(False, False, 0.0, 0.0, 0.0, "Frames unreadable")

    # 3) Analizar frames
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

            sh = sharpness(img)   # nitidez
            br = brightness(img)  # brillo

            analyzed.append({
            "img": img,
            "face_ok": face_ok,
            "glasses": wears,
            "sharp": sh,
            "bright": br,
            })

        if valid_faces > 0 and glasses_all_valid:
            return EvaluationResult(False, False, 0.0, 0.0, 0.0, "Glasses in all valid frames")

    # 4) Reto mínimo (placeholder: sustituir con EAR/MAR + yaw/pitch/roll reales)
        challenge = 0.92 if valid_faces >= 10 else 0.6

    # 5) Selección de best frame (mezcla nitidez + brillo + rostro)
    #    - Normaliza nitidez con p90 (robusto) y equilibra brillo hacia ~128
        W_SHARP = 0.55
        W_BRIGHT = 0.30
        W_FACE_BONUS = 0.15

        sharp_values = [f["sharp"] for f in analyzed]
        p90 = np.percentile(sharp_values, 90) if sharp_values else 500.0
        p90 = max(p90, 1.0)

        def score_frame(f):
            sharp_norm = min(f["sharp"] / p90, 1.0)              # 0..1
            bright_score = max(0.0, 1.0 - abs((f["bright"] - 128.0) / 128.0))  # 0..1
            face_term = (W_FACE_BONUS if f["face_ok"] else -W_FACE_BONUS)
            return (W_SHARP * sharp_norm) + (W_BRIGHT * bright_score) + face_term

        candidates = [f for f in analyzed if f["face_ok"]] or analyzed
        best = max(candidates, key=score_frame)
        best_img = best["img"]

    # 6) Anti-spoof + Luxand (opcional)
        spoof_prob = self.antispoof.spoof_probability(best_img)   # 0..1 (0=real)
        antispoof_component = 1.0 - spoof_prob                    # 0..1
        luxand = self.liveness.score(best_img)                    # 0..1

    # ¿Luxand activo? (si tu cliente real expone is_active; si no, usa luxand>0.0)
        luxand_active_attr = getattr(self.liveness, "is_active", None)
        luxand_active = (bool(luxand_active_attr) if luxand_active_attr is not None else (luxand > 0.0))

        if luxand_active:
        # Pesos con Luxand
            live_score = 0.45*challenge + 0.35*antispoof_component + 0.20*luxand
            liveness_ok = (live_score >= self.t.live) and (luxand >= self.t.luxand)
        else:
        # Sin Luxand (Noop): repondera y no exige umbral de Luxand
            live_score = 0.60*challenge + 0.40*antispoof_component
            liveness_ok = (live_score >= self.t.live)

    # 7) Similaridad (AWS Rekognition)
        similarity = self.matcher.compare(best_img, ref_img)  # 0..100
        match_ok = (similarity >= self.t.similarity)

    # 8) Evaluación 0..100 (50% liveness, 50% similarity)
        evaluation_pct = 0.5*(live_score*100.0) + 0.5*similarity
        passed = liveness_ok and match_ok and (evaluation_pct >= 95.0)

        msg = "OK" if passed else (
        "Liveness insuficiente" if not liveness_ok else
        ("Baja similitud" if not match_ok else "Evaluación < 95%")
        )
        return EvaluationResult(liveness_ok, match_ok, live_score, similarity, evaluation_pct, msg)
