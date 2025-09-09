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
            confidence = float(det.get("confidence", 0.0)) if face_ok else 0.0
            area_rel  = float(det.get("area_rel", 0.0)) if face_ok else 0.0
            pose      = det.get("pose", {}) if face_ok else {}
            quality   = det.get("quality", {}) if face_ok else {}

            if face_ok:
                valid_faces += 1
                if wears:
                    valid_with_glasses += 1

            analyzed.append({
                "path": path,
                "img": img,
                "face_ok": face_ok,
                "bbox": bbox,
                "glasses": wears,
                "sharp_full": sharpness(img),     # por si quieres depurar
                "bright_full": brightness(img),
                "confidence": confidence,
                "area_rel": area_rel,
                "pose": pose,
                "quality": quality,
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
        def crop_bbox(img_bgr, bbox, pad_ratio=0.2):
            if bbox is None: return img_bgr
            h, w = img_bgr.shape[:2]
            x1, y1, x2, y2 = bbox
            bw, bh = x2 - x1, y2 - y1
            px = int(bw * pad_ratio); py = int(bh * pad_ratio)
            X1 = max(0, x1 - px); Y1 = max(0, y1 - py)
            X2 = min(w, x2 + px); Y2 = min(h, y2 + py)
            if X2 <= X1 or Y2 <= Y1: return img_bgr
            return img_bgr[Y1:Y2, X1:X2]
        def face_laplacian_sharp(face_bgr) -> float:
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())
        # Normalizaciones ABSOLUTAS
        def norm_sharp_rek(q_sharp):   # Rekognition Quality.Sharpness 0..100
            # 80+ ya es muy nítido, satura a 1.0
            return max(0.0, min(q_sharp / 80.0, 1.0))
        
        def norm_sharp_lap(sh):        # Laplaciano (fallback). 150 es muy nítido
            return max(0.0, min(sh / 150.0, 1.0))

        def norm_bright_rek(q_bright): # Rekognition 0..100; ideal ~50
            return max(0.0, 1.0 - abs((q_bright - 50.0) / 50.0))

        def norm_area_rel(area_rel):   # ideal ≥ 25%
            return max(0.0, min(area_rel / 0.25, 1.0))

        def norm_conf(conf):           # 0..100 → 0..1
            return max(0.0, min(conf / 100.0, 1.0))

        def frontalness(yaw, roll):
            # 1.0 cuando |yaw|,|roll|≈0; penaliza > 20°. Saturación a 45°.
            yaw_pen = min(abs(float(yaw)), 45.0) / 45.0  # 0..1
            roll_pen = min(abs(float(roll)), 45.0) / 45.0
            # mezcla (yaw pesa más que roll)
            return max(0.0, 1.0 - (0.7 * yaw_pen + 0.3 * roll_pen))
        
        W_FRONTAL = 0.35
        W_SHARP   = 0.30
        W_AREA    = 0.20
        W_CONF    = 0.10
        W_BRIGHT  = 0.05
        candidates = []

        for f in analyzed:
            if not f["face_ok"]:
                continue
            bbox = f["bbox"]
            face_crop = crop_bbox(f["img"], bbox, pad_ratio=0.2)

            # Calidad: usa Quality de Rekognition si existe; si no, Laplaciano
            q = f.get("quality") or {}
            q_sharp = float(q.get("Sharpness", 0.0))
            q_bright = float(q.get("Brightness", 50.0))  # centro 50

            sharp_norm = norm_sharp_rek(q_sharp) if q else norm_sharp_lap(face_laplacian_sharp(face_crop))
            bright_norm = norm_bright_rek(q_bright)
            area_norm = norm_area_rel(float(f.get("area_rel", 0.0)))
            conf_norm = norm_conf(float(f.get("confidence", 0.0)))
            pose = f.get("pose") or {}
            yaw = float(pose.get("Yaw", 0.0)); roll = float(pose.get("Roll", 0.0))
            frontal = frontalness(yaw, roll)

            score = (W_FRONTAL * frontal) + (W_SHARP * sharp_norm) + (W_AREA * area_norm) + (W_CONF * conf_norm) + (W_BRIGHT * bright_norm)

            candidates.append({
                **f,
                "face_crop": face_crop,
                "sharp_face_norm": round(sharp_norm, 4),
                "bright_face_norm": round(bright_norm, 4),
                "area_rel": float(f.get("area_rel", 0.0)),
                "confidence": float(f.get("confidence", 0.0)),
                "yaw": yaw, "roll": roll,
                "frontal": round(frontal, 4),
                "score": round(score, 4),
            })

        if not candidates:
            logger.info({"uuid": uuid_proceso, "event": "no_face_candidates_for_best"})
            return EvaluationResult(False, False, 0.0, 0.0, 0.0, "No face candidates")

        # Ordena y loggea Top-3 para auditar por qué ganó
        candidates.sort(key=lambda x: x["score"], reverse=True)
        topk = candidates[:3]
        logger.info({
            "uuid": uuid_proceso,
            "event": "best_frame_topk",
            "topk": [
                {
                    "path": t["path"],
                    "score": t["score"],
                    "frontal": t["frontal"],
                    "yaw": round(t["yaw"], 1), "roll": round(t["roll"], 1),
                    "sharp_norm": t["sharp_face_norm"],
                    "bright_norm": t["bright_face_norm"],
                    "area_rel_pct": round(t["area_rel"]*100.0, 2),
                    "conf": round(t["confidence"], 1),
                } for t in topk
            ]
        })

        best = candidates[0]
        best_img = best["img"]
        best_path = best.get("path", "<unknown>")

        logger.info({
            "uuid": uuid_proceso,
            "event": "best_frame",
            "path": best_path,
            "score": best["score"],
            "frontal": best["frontal"],
            "yaw": round(best["yaw"], 1), "roll": round(best["roll"], 1),
            "sharp_norm": best["sharp_face_norm"],
            "bright_norm": best["bright_face_norm"],
            "area_rel_pct": round(best["area_rel"]*100.0, 2),
            "conf": round(best["confidence"], 1)
        })

        

        # 6) Anti-spoof + Luxand (usar IMAGEN COMPLETA o face_crop según prefieras)
        # --- Anti-spoof ---
        spoof_prob = self.antispoof.spoof_probability(best_img)   # 0..1 (0=real)
        antispoof_component = 1.0 - spoof_prob                    # 0..1

        # --- Luxand ---
        # Intentamos una interfaz rica: evaluate() -> {'ok':bool,'score':float,'label':str|None}
        lux_ok, lux_score, lux_label = False, 0.0, None
        try:
            res = self.liveness.evaluate(best_img)  # si tu implementación la tiene
            if isinstance(res, dict):
                lux_ok = bool(res.get("ok", False))
                lux_score = float(res.get("score", 0.0) or 0.0)
                lux_label = res.get("label")
            else:
                # fallback por si evaluate() devuelve tuple
                lux_ok, lux_score = bool(res[0]), float(res[1])
                lux_label = res[2] if len(res) > 2 else None
        except AttributeError:
            # compatibilidad con la API actual score() -> 0..1
            try:
                lux_score = float(self.liveness.score(best_img) or 0.0)
                lux_ok = lux_score > 0.0
            except Exception:
                lux_score = 0.0
                lux_ok = False

        # Determinar si Luxand está "activo"
        luxand_active_attr = getattr(self.liveness, "is_active", None)
        if callable(luxand_active_attr):
            luxand_active = bool(luxand_active_attr())
        elif luxand_active_attr is not None:
            luxand_active = bool(luxand_active_attr)
        else:
            luxand_active = lux_ok or (lux_score > 0.0)

        # --- REGLA DE DECISIÓN ---
        # 1) Short-circuit: si Luxand pasa solo, aprueba liveness
        luxand_passes = luxand_active and lux_ok and (lux_score >= self.t.luxand) and (lux_label in (None, "real"))
        if luxand_passes:
            live_score = max(lux_score, 0.90)  # opcional: asegura un mínimo alto cuando Luxand afirma REAL
            liveness_ok = True
        else:
            # 2) Ponderación (ajusta pesos/umbral si tu política lo permite)
            W_CHALLENGE = 0.20
            W_ANTISPOOF = 0.20
            W_LUXAND    = 0.60
            live_score = (W_CHALLENGE * challenge) + (W_ANTISPOOF * antispoof_component) + (W_LUXAND * lux_score)
            # Si Luxand está activo, exige ambos; si no, solo el score total
            liveness_ok = (live_score >= getattr(self.t, "live", 0.85)) and (not luxand_active or lux_score >= self.t.luxand)

        logger.info({
            "uuid": uuid_proceso,
            "event": "liveness_components",
            "challenge": round(challenge, 4),
            "challenge_pct": round(challenge * 100.0, 2),
            "antispoof_prob_spoof": round(spoof_prob, 4),
            "antispoof_component_real": round(antispoof_component, 4),
            "antispoof_component_real_pct": round(antispoof_component * 100.0, 2),
            "luxand_score": round(lux_score, 4),
            "luxand_score_pct": round(lux_score * 100.0, 2),
            "luxand_label": lux_label,
            "luxand_active": luxand_active,
            "liveness_score": round(live_score, 4),
            "liveness_score_pct": round(live_score * 100.0, 2),
            "threshold_live": getattr(self.t, "live", 0.85),
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

        # 8) evaluación por promedio simple de filtros (con normalizaciones y pisos si Luxand=REAL)
        filters = []

        # --- valores base para evaluación (0..1) ---
        # challenge y antispoof_component ya están 0..1
        challenge_for_eval = float(challenge)
        antispoof_for_eval = float(antispoof_component)

        # Asegura que tenemos el lux_score real (0..1) para el promedio
        try:
            lux_score_eval = float(lux_score)
        except Exception:
            lux_score_eval = 0.0

        # Normaliza área del rostro con meta 25% como 1.0
        # (si area_rel ya viene en 0..1 del encuadre)
        area_rel_raw = float(best.get("area_rel", 0.0))  # 0..1
        area_rel_eval = min(area_rel_raw / 0.25, 1.0)    # >=25% satura en 1.0

        # Si Luxand afirmó REAL (alto score), aplica pisos mínimos razonables a heurísticos
        if luxand_active and (lux_score_eval >= self.t.luxand) and (lux_label in (None, "real")):
            # estos pisos solo afectan la EVALUACIÓN (no tu liveness interno)
            challenge_for_eval   = max(challenge_for_eval, 0.95)   # 95 puntos mínimo
            antispoof_for_eval   = max(antispoof_for_eval, 0.90)   # 90 puntos mínimo

        # Construimos los filtros (todos en 0..100) y promediamos
        # 1) challenge
        filters.append(("challenge", challenge_for_eval * 100.0))

        # 2) antispoof
        filters.append(("antispoof", antispoof_for_eval * 100.0))

        # 3) luxand (si está activo y >0)
        if luxand_active and lux_score_eval > 0.0:
            filters.append(("luxand", lux_score_eval * 100.0))

        # 4) similarity (ya es 0..100)
        filters.append(("similarity", float(similarity)))

        # 5) frontal (0..1 -> 0..100)
        filters.append(("frontal", float(best.get("frontal", 0.0)) * 100.0))

        # 6) sharp_norm (0..1 -> 0..100)
        filters.append(("sharp_norm", float(best.get("sharp_face_norm", 0.0)) * 100.0))

        # 7) area_rel normalizada a meta 25% (0..1 -> 0..100)
        filters.append(("area_rel", area_rel_eval * 100.0))

        # Si quieres también brillo normalizado, descomenta:
        # filters.append(("bright_norm", float(best.get("bright_face_norm", 0.0)) * 100.0))

        # Promedio simple
        scores_only = [v for _, v in filters]
        evaluation_pct = sum(scores_only) / max(len(scores_only), 1)

        # Umbral único
        EVAL_THRESHOLD = 95.0
        passed = (evaluation_pct >= EVAL_THRESHOLD)

        msg = "OK" if passed else f"Evaluación < {EVAL_THRESHOLD}%"

        logger.info({
            "uuid": uuid_proceso,
            "event": "final_decision",
            "status": "success" if passed else "false",
            "message": msg,
            "evaluation_pct": round(evaluation_pct, 2),
            "filters": [{"name": k, "value": round(v, 2)} for k, v in filters],
            "filters_count": len(filters),
            # auditoría: flags previos
            "liveness_ok": liveness_ok,
            "match_ok": match_ok,
            "luxand_passes": bool(luxand_active and (lux_score_eval >= self.t.luxand) and (lux_label in (None, "real")))
        })

        return EvaluationResult(liveness_ok, match_ok, live_score, similarity, evaluation_pct, msg)

        