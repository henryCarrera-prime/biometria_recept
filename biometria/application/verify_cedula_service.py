# biometria/application/verify_cedula_service.py
import base64
import uuid
import cv2
import numpy as np
import os
import re
from dataclasses import dataclass
from typing import Dict, Any, Protocol, Optional, Tuple, Iterable

from biometria.domain.interfaces import (
    FaceDetector,
    ReferenceImageRepository,
    SimilarityMatcher,
)

# --- Helpers ENV robustos (soportan "95.0 # comentario") ---
def _env_float(var: str, default: float) -> float:
    raw = os.getenv(var, str(default))
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(raw))
    return float(m.group(0)) if m else float(default)

def _env_str(var: str, default: str) -> str:
    return str(os.getenv(var, default)).strip()

def _env_int_list(var: str, default: Iterable[int]) -> set[int]:
    raw = _env_str(var, ",".join(str(x) for x in default))
    out: set[int] = set()
    for token in str(raw).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        m = re.search(r"-?\d+", token)
        if m:
            out.add(int(m.group(0)))
    return out

EVALUATION_THRESHOLD = _env_float("ECU_ID_EVALUATION_THRESHOLD", 95.0)
# Umbral mínimo del score del modelo (0..1)
ECU_ID_THRESHOLD = _env_float("ECU_ID_THRESHOLD", 0.85)
# Índices del modelo que se consideran "cédula aprobada"
POSITIVE_LABELS = _env_int_list("ECU_ID_POSITIVE_LABELS", default=[1, 2, 3])

# Si algún día quieres volver a comparar por rostro, cambia a 'face'
COMPARE_MODE = os.getenv("ECU_ID_COMPARE_MODE", "full").lower()  # 'full' | 'face'

class _IdClassifier(Protocol):
    def is_valid_ec_id(self, image_bgr: np.ndarray) -> Tuple[bool, float]:
        """
        Retorna:
          - bool: si el modelo considera que es cédula válida (según su threshold interno)
          - float: score 0..1
        """
        ...
    # Opcional: método para obtener el label
    # def predict_label(self, image_bgr: np.ndarray) -> int: ...

def _try_get_predicted_label(classifier: Any, image_bgr: np.ndarray) -> Optional[int]:
    """Intenta obtener el label predicho del classifier (duck-typed)."""
    for attr in ("predict_label", "predict_class", "classify", "predict"):
        fn = getattr(classifier, attr, None)
        if callable(fn):
            try:
                pred = fn(image_bgr)
                if isinstance(pred, (list, tuple)) and len(pred) >= 1:
                    lbl = pred[0]
                    return int(lbl) if lbl is not None else None
                if isinstance(pred, dict):
                    if "label" in pred:
                        return int(pred["label"])
                    if "class" in pred:
                        return int(pred["class"])
                try:
                    return int(pred)  # e.g. np.int
                except Exception:
                    pass
            except Exception:
                return None
    for prop in ("last_label", "last_class", "predicted_label"):
        if hasattr(classifier, prop):
            try:
                return int(getattr(classifier, prop))
            except Exception:
                pass
    return None

def _b64_to_bgr(b64: str) -> np.ndarray:
    data = base64.b64decode(b64.split(",")[-1])
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("imageBase64 inválido (no se pudo decodificar)")
    return img

def _clip_bbox(bbox, w, h):
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)

@dataclass
class EcuadorIdVerificationResponse:
    status: bool
    message: str
    payload: Dict[str, Any]
    diagnostics: Optional[Dict[str, Any]] = None

class VerifyEcuadorIdService:
    def __init__(
        self,
        id_classifier: _IdClassifier,
        face_detector: FaceDetector,
        ref_repo: ReferenceImageRepository,
        similarity: SimilarityMatcher,
    ):
        self.id_classifier = id_classifier
        self.face_detector = face_detector
        self.ref_repo = ref_repo
        self.similarity = similarity

    def execute(self, uuid_proceso: str, image_b64: str, reference_url: str) -> EcuadorIdVerificationResponse:
        uuid_proceso_cedula = str(uuid.uuid4())

        id_img_bgr = _b64_to_bgr(image_b64)
        H, W = id_img_bgr.shape[:2]

        # 1) Clasificación de cédula (0..1) -> %
        is_valid_id_raw, id_score = self.id_classifier.is_valid_ec_id(id_img_bgr)
        id_score_pct = float(id_score) * 100.0

        # 1.1) Label predicho (OBLIGATORIO) y permitidos desde ENV
        predicted_label = _try_get_predicted_label(self.id_classifier, id_img_bgr)
        label_ok = (predicted_label is not None) and (predicted_label in POSITIVE_LABELS)
        score_ok = (id_score >= ECU_ID_THRESHOLD)

        # Cédula válida final: modelo OK AND label permitido AND score suficiente
        is_valid_id_final = bool(is_valid_id_raw and label_ok and score_ok)

        # --- EARLY RETURN para ahorrar costo: si NO es cédula, NO llamamos a Rekognition ---
        if not is_valid_id_final:
            # Intento de detección solo para diagnóstico (no afecta respuesta)
            diag = {
                "id_score_pct": round(id_score_pct, 2),
                "similarity_pct": None,  # NO calculada
                "threshold_eval_pct": EVALUATION_THRESHOLD,
                "model_threshold": ECU_ID_THRESHOLD,
                "failure_reason": None,
                "face_bbox_id": None,
                "face_bbox_ref": None,
                "compare_mode_used": "skipped",  # saltamos similitud
                "predicted_label": predicted_label,
                "positive_labels_env": sorted(list(POSITIVE_LABELS)),
                "label_ok": label_ok,
                "score_ok": score_ok,
                "is_valid_id_raw": bool(is_valid_id_raw),
                "is_valid_id_final": False,
            }
            try:
                fi = self.face_detector.detect(id_img_bgr)
                if fi and fi.get("bbox"):
                    b = _clip_bbox(fi["bbox"], W, H)
                    if b:
                        diag["face_bbox_id"] = [int(v) for v in b]
            except Exception:
                pass

            # Mensaje preciso de por qué no es cédula
            if predicted_label is None:
                diag["failure_reason"] = "no_label"
                message = "La imagen NO corresponde a una cédula válida (el modelo no pudo determinar una clase de cédula)."
            elif not label_ok:
                diag["failure_reason"] = "label_not_allowed"
                message = f"La imagen NO corresponde a una cédula válida: label {predicted_label} no está en {sorted(list(POSITIVE_LABELS))}."
            elif not score_ok:
                diag["failure_reason"] = "low_model_score"
                message = f"La imagen NO corresponde a una cédula válida: score del modelo {id_score:.3f} < umbral {ECU_ID_THRESHOLD:.3f}."
            else:
                diag["failure_reason"] = "invalid_id_generic"
                message = "La imagen NO corresponde a una cédula válida."

            # Evaluación informativa SOLO del componente del modelo (no hay similitud)
            evaluacion = 0.0

            return EcuadorIdVerificationResponse(
                status=False,
                message=message,
                payload={
                    "uuidProceso": uuid_proceso,
                    "data": [{"uuid_proceso_cedula": uuid_proceso_cedula, "evaluacion": evaluacion}],
                },
                diagnostics=diag,
            )

        # --- Si es cédula válida, continuamos con referencia + similitud ---
        ref_img = self.ref_repo.fetch_reference(reference_url)
        if ref_img is None:
            evaluacion = round(id_score_pct, 2)  # solo el componente de cédula
            return EcuadorIdVerificationResponse(
                status=False,
                message="No se pudo obtener la imagen de referencia.",
                payload={
                    "uuidProceso": uuid_proceso,
                    "data": [{"uuid_proceso_cedula": uuid_proceso_cedula, "evaluacion": evaluacion}],
                },
                diagnostics={
                    "id_score_pct": round(id_score_pct, 2),
                    "similarity_pct": None,
                    "threshold_eval_pct": EVALUATION_THRESHOLD,
                    "model_threshold": ECU_ID_THRESHOLD,
                    "failure_reason": "no_reference_image",
                    "compare_mode_used": "full",
                    "predicted_label": predicted_label,
                    "positive_labels_env": sorted(list(POSITIVE_LABELS)),
                    "label_ok": True,
                    "score_ok": True,
                    "is_valid_id_raw": True,
                    "is_valid_id_final": True,
                }
            )

        # 3) Detección (solo diagnóstico)
        diag = {
            "id_score_pct": round(id_score_pct, 2),
            "similarity_pct": None,
            "threshold_eval_pct": EVALUATION_THRESHOLD,
            "model_threshold": ECU_ID_THRESHOLD,
            "failure_reason": None,
            "face_bbox_id": None,
            "face_bbox_ref": None,
            "compare_mode_used": "full",
            "predicted_label": predicted_label,
            "positive_labels_env": sorted(list(POSITIVE_LABELS)),
            "label_ok": True,
            "score_ok": True,
            "is_valid_id_raw": True,
            "is_valid_id_final": True,
        }

        try:
            fi = self.face_detector.detect(id_img_bgr)
            if fi and fi.get("bbox"):
                b = _clip_bbox(fi["bbox"], W, H)
                if b:
                    diag["face_bbox_id"] = [int(v) for v in b]
        except Exception:
            pass

        try:
            Hr, Wr = ref_img.shape[:2]
            fr = self.face_detector.detect(ref_img)
            if fr and fr.get("bbox"):
                rb = _clip_bbox(fr["bbox"], Wr, Hr)
                if rb:
                    diag["face_bbox_ref"] = [int(v) for v in rb]
        except Exception:
            pass

        # 4) Similaridad (0..100) — ahora sí consume AWS Rekognition
        try:
            similarity_pct = float(self.similarity.compare(id_img_bgr, ref_img))
        except Exception:
            similarity_pct = 0.0
        diag["similarity_pct"] = round(similarity_pct, 2)

        # 5) Evaluación final (promedio) y regla de éxito
        evaluacion = (id_score_pct + similarity_pct) / 2.0
        passed = (evaluacion > EVALUATION_THRESHOLD)

        message = (
            "Cédula válida y rostro coincide con la imagen de referencia."
            if passed else
            "Cédula válida pero el rostro NO coincide con la imagen de referencia."
        )

        return EcuadorIdVerificationResponse(
            status=passed,
            message=message,
            payload={
                "uuidProceso": uuid_proceso,
                "data": [{"uuid_proceso_cedula": uuid_proceso_cedula, "evaluacion": round(evaluacion, 2)}],
            },
            diagnostics=diag,
        )
