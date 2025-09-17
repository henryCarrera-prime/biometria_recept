# biometria/application/verify_cedula_service.py
import base64
import uuid
import cv2
import numpy as np
import os
import re
from dataclasses import dataclass
from typing import Dict, Any, Protocol, Optional

from biometria.domain.interfaces import (
    FaceDetector,
    ReferenceImageRepository,
    SimilarityMatcher,
)

# --- ENV robusto (soporta "95.0 # comentario") ---
def _env_float(var: str, default: float) -> float:
    raw = os.getenv(var, str(default))
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(raw))
    return float(m.group(0)) if m else float(default)

EVALUATION_THRESHOLD = _env_float("ECU_ID_EVALUATION_THRESHOLD", 95.0)

# Si algún día quieres volver a comparar por rostro, cambia a 'face'
COMPARE_MODE = os.getenv("ECU_ID_COMPARE_MODE", "full").lower()  # 'full' | 'face'

class _IdClassifier(Protocol):
    def is_valid_ec_id(self, image_bgr: np.ndarray) -> tuple[bool, float]:
        ...

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

        # 1) Validación cédula (0..1) -> %
        is_valid_id, id_score = self.id_classifier.is_valid_ec_id(id_img_bgr)
        id_score_pct = float(id_score) * 100.0

        # 2) Obtener referencia (OBLIGATORIO para comparar)
        ref_img = self.ref_repo.fetch_reference(reference_url)
        if ref_img is None:
            evaluacion = (id_score_pct + 0.0) / 2.0
            return EcuadorIdVerificationResponse(
                status=False,
                message="No se pudo obtener la imagen de referencia.",
                payload={
                    "uuidProceso": uuid_proceso,
                    "data": [{"uuid_proceso_cedula": uuid_proceso_cedula, "evaluacion": round(evaluacion, 2)}],
                },
                diagnostics={
                    "id_score_pct": round(id_score_pct, 2),
                    "similarity_pct": None,
                    "threshold_eval_pct": EVALUATION_THRESHOLD,
                    "failure_reason": "no_reference_image",
                    "compare_mode_used": "full",
                }
            )

        # 3) Detección solo para diagnóstico (NO recortamos)
        diag = {
            "id_score_pct": round(id_score_pct, 2),
            "similarity_pct": None,
            "threshold_eval_pct": EVALUATION_THRESHOLD,
            "failure_reason": None,
            "face_bbox_id": None,
            "face_bbox_ref": None,
            "compare_mode_used": "full",  # vamos a comparar imagen completa
        }

        # Intentamos detectar para loguear (no afecta la comparación)
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

        # 4) Similaridad (0..100) usando IMAGEN COMPLETA SIEMPRE
        try:
            similarity_pct = float(self.similarity.compare(id_img_bgr, ref_img))
        except Exception:
            similarity_pct = 0.0

        diag["similarity_pct"] = round(similarity_pct, 2)

        # 5) Evaluación (promedio)
        evaluacion = (id_score_pct + similarity_pct) / 2.0
        passed = (evaluacion > EVALUATION_THRESHOLD) and is_valid_id

        message = (
            "La imagen NO corresponde a una cédula válida."
            if not is_valid_id else
            ("Cédula válida y rostro coincide con la imagen de referencia." if passed
             else "Cédula válida pero el rostro NO coincide con la imagen de referencia.")
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
