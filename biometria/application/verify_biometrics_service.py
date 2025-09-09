# biometria/application/verify_biometrics_service.py
from typing import List
from ..domain.value_objects import Thresholds, EvaluationResult
from ..domain.interfaces import (
    ReferenceImageRepository, FramesRepository, FaceDetector,
    GlassesClassifier, AntiSpoof, LivenessPassive, SimilarityMatcher
)
import cv2, numpy as np
import logging
logger = logging.getLogger("biometria.verify")

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

        # ---- helpers ----
        def sharpness(img_bgr) -> float:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())

        def brightness(img_bgr) -> float:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            return float(np.mean(gray))

        def crop_bbox(img_bgr, bbox, pad_ratio=0.2):
            """
            bbox = (x1,y1,x2,y2) en pixeles. Aplica padding y recorta con límites.
            """
            if bbox is None:
                return img_bgr
            h, w = img_bgr.shape[:2]
            x1, y1, x2, y2 = bbox
            bw, bh = x2 - x1, y2 - y1
            px = int(bw * pad_ratio)
            py = int(bh * pad_ratio)
            X1 = max(0, x1 - px); Y1 = max(0, y1 - py)
            X2 = min(w, x2 + px); Y2 = min(h, y2 + py)
            if X2 <= X1 or Y2 <= Y1:
                return img_bgr
            return img_bgr[Y1:Y2, X1:X2]

        # 1) referencia
        ref_img = self.reference_repo.fetch_reference(reference_dir)
        if ref_img is None:
            logger.info({"uuid": uuid_proceso, "event": "reference_not_found", "reference_dir": reference_dir})
            return EvaluationResult(False, False, 0.0, 0.0, 0.0, "Reference not found")

        # 2) frames
        frame_paths = self.frames_repo.list_frames(frames_dir)
        if not frame_paths:
            logger.info({"uuid": uuid_proceso, "event": "no_frames_found", "frames_dir": frames_dir})
            return EvaluationResult(False, False, 0.0, 0.0, 0.0, "No frames found")

        imgs = [self.frames_repo.fetch_frame(p) for p in frame_paths]
        pairs = [(p, im) for p, im in zip(frame_paths, imgs) if im is not None]
        if not pairs:
            logger.info({"uuid": uuid_proceso, "event": "frames_unreadable", "frames_dir": frames_dir})
            return EvaluationResult(False, False, 0.0, 0.0, 0.0, "Frames unreadable")

        # 3) análisis frame a frame (GUARDA bbox)
        analyzed = []
        valid_faces = 0
        valid_with_glasses = 0

        for path, img in pairs:
            det = self.detector.detect(img)  # dict or None. Esperamos {"bbox":(x1,y1,x2,y2), "landmarks":...}
            face_ok = det is not None
            bbox = det.get("bbox") if det else None
            wears = self.glasses.has_glasses(img, det.get("landmarks")) if face_ok else False

            if face_ok:
                valid_faces += 1
                if wears:
                    valid_with_glasses += 1

            analyzed.append({
                "path": path,
                "img": img,
                "face_ok": face_ok,
                "bbox": bbox,              # <<<<<< guardamos bbox
                "glasses": wears,
                "sharp": sharpness(img),
                "bright": brightness(img),
            })

        total_frames = len(analyzed)
        pct_face = (valid_faces / total_frames) * 100.0
        valid_without_glasses = max(valid_faces - valid_with_glasses, 0)
        pct_valid_with_glasses = (valid_with_glasses / valid_faces) * 100.0 if valid_faces else 0.0
        pct_valid_without_glasses = (valid_without_glasses / valid_faces) * 100.0 if valid_faces else 0.0

        logger.info({
            "uuid": uuid_proceso,
            "event": "frames_stats",
            "frames_dir": frames_dir,
            "total_frames": total_frames,
            "valid_faces": valid_faces,
            "pct_frames_with_face": round(pct_face, 2),
            "valid_with_glasses": valid_with_glasses,
            "valid_without_glasses": valid_without_glasses,
            "pct_valid_with_glasses": round(pct_valid_with_glasses, 2),
            "pct_valid_without_glasses": round(pct_valid_without_glasses, 2)
        })

        # 3.1) si NO hay ningún rostro, no sigas (evita falsos 0.0 de Luxand)
        if valid_faces == 0:
            logger.info({"uuid": uuid_proceso, "event": "no_face_in_frames"})
            return EvaluationResult(False, False, 0.0, 0.0, 0.0, "No face detected in frames")

        # 3.2) regla de lentes: si TODOS los válidos tienen lentes → falla
        if valid_with_glasses == valid_faces:
            logger.info({"uuid": uuid_proceso, "event": "fail_glasses_all_valid"})
            return EvaluationResult(False, False, 0.0, 0.0, 0.0, "Glasses in all valid frames")

        # 4) reto / movimiento (placeholder)
        challenge = 0.92 if valid_faces >= 10 else 0.6
        logger.info({
            "uuid": uuid_proceso,
            "event": "challenge_estimate",
            "kind": "heuristic",
            "challenge": round(challenge, 4),
            "challenge_pct": round(challenge * 100.0, 2)
        })

        # 5) best frame (nitidez + brillo + rostro)
        W_SHARP = 0.55
        W_BRIGHT = 0.30
        W_FACE_BONUS = 0.15

        sharp_values = [f["sharp"] for f in analyzed]
        p90 = np.percentile(sharp_values, 90) if sharp_values else 500.0
        p90 = max(p90, 1.0)

        def score_frame(f):
            sharp_norm = min(f["sharp"] / p90, 1.0)
            bright_score = max(0.0, 1.0 - abs((f["bright"] - 128.0) / 128.0))
            face_term = (W_FACE_BONUS if f["face_ok"] else -W_FACE_BONUS)
            return (W_SHARP * sharp_norm) + (W_BRIGHT * bright_score) + face_term

        candidates = [f for f in analyzed if f["face_ok"]]  # <-- ahora sí, EXIGIMOS rostro
        best = max(candidates, key=score_frame)
        best_img = best["img"]
        best_path = best.get("path", "<unknown>")
        best_bbox = best.get("bbox")  # bbox del rostro en best

        logger.info({
            "uuid": uuid_proceso,
            "event": "best_frame",
            "path": best_path,
            "face_ok": best["face_ok"],
            "glasses": best["glasses"],
            "sharp": round(best["sharp"], 2),
            "bright": round(best["bright"], 2),
            "score": round(score_frame(best), 4),
            "candidates": len(candidates),
            "total_frames": total_frames
        })

        # 6) Anti-spoof + Luxand (usar CROP de rostro para Luxand)
        spoof_prob = self.antispoof.spoof_probability(best_img)   # 0..1 (0=real)
        antispoof_component = 1.0 - spoof_prob                    # 0..1

        # recorte del rostro para liveness
        face_for_liveness = crop_bbox(best_img, best_bbox, pad_ratio=0.2)
        luxand = self.liveness.score(face_for_liveness)           # 0..1

        luxand_active_attr = getattr(self.liveness, "is_active", None)
        luxand_active = (bool(luxand_active_attr) if luxand_active_attr is not None else (luxand > 0.0))

        if luxand_active:
            live_score = 0.45*challenge + 0.35*antispoof_component + 0.20*luxand
            liveness_ok = (live_score >= self.t.live) and (luxand >= self.t.luxand)
        else:
            live_score = 0.60*challenge + 0.40*antispoof_component
            liveness_ok = (live_score >= self.t.live)

        logger.info({
            "uuid": uuid_proceso,
            "event": "liveness_components",
            "challenge": round(challenge, 4),
            "challenge_pct": round(challenge * 100.0, 2),
            "antispoof_prob_spoof": round(spoof_prob, 4),
            "antispoof_component_real": round(antispoof_component, 4),
            "antispoof_component_real_pct": round(antispoof_component * 100.0, 2),
            "luxand_score": round(luxand, 4),
            "luxand_score_pct": round(luxand * 100.0, 2),
            "luxand_active": luxand_active,
            "used_face_crop_for_luxand": best_bbox is not None,
            "liveness_score": round(live_score, 4),
            "liveness_score_pct": round(live_score * 100.0, 2),
            "threshold_live": self.t.live,
            "threshold_luxand": self.t.luxand if luxand_active else None,
            "liveness_ok": liveness_ok
        })

        # 7) Similaridad (AWS)
        similarity = self.matcher.compare(best_img, ref_img)  # 0..100
        match_ok = (similarity >= self.t.similarity)

        logger.info({
            "uuid": uuid_proceso,
            "event": "similarity",
            "best_frame_path": best_path,
            "similarity": round(similarity, 2),
            "threshold_similarity": self.t.similarity,
            "match_ok": match_ok
        })

        # 8) evaluación 0..100
        evaluation_pct = 0.5*(live_score*100.0) + 0.5*similarity
        passed = liveness_ok and match_ok and (evaluation_pct >= 95.0)
        msg = "OK" if passed else ("Liveness insuficiente" if not liveness_ok else ("Baja similitud" if not match_ok else "Evaluación < 95%"))

        logger.info({
            "uuid": uuid_proceso,
            "event": "final_decision",
            "status": "success" if passed else "false",
            "message": msg,
            "evaluation_pct": round(evaluation_pct, 2),
            "liveness_ok": liveness_ok,
            "match_ok": match_ok
        })

        return EvaluationResult(liveness_ok, match_ok, live_score, similarity, evaluation_pct, msg)


        