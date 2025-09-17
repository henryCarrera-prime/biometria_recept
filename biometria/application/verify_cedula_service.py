# biometria/application/verify_cedula_service.py
import base64
import uuid
import cv2
import numpy as np
import os
from dataclasses import dataclass
from typing import Dict, Any, Protocol  # <- para un Protocol local

from biometria.domain.interfaces import (
    FaceDetector,
    ReferenceImageRepository,
    SimilarityMatcher,   # <- usa tu protocolo real de dominio
)

EVALUATION_THRESHOLD = float(os.getenv("ECU_ID_EVALUATION_THRESHOLD", "95.0"))

# Protocol local SOLO para typing (no toca tu domain)
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

@dataclass
class EcuadorIdVerificationResponse:
    status: bool
    message: str
    payload: Dict[str, Any]  # incluye uuidProceso + data[]

class VerifyEcuadorIdService:
    def __init__(
        self,
        id_classifier: _IdClassifier,               # <- NO depende de domain
        face_detector: FaceDetector,
        ref_repo: ReferenceImageRepository,
        similarity: SimilarityMatcher,              # <- usa tu SimilarityMatcher
    ):
        self.id_classifier = id_classifier
        self.face_detector = face_detector
        self.ref_repo = ref_repo
        self.similarity = similarity

    def execute(self, uuid_proceso: str, image_b64: str, reference_url: str) -> EcuadorIdVerificationResponse:
        uuid_proceso_cedula = str(uuid.uuid4())

        id_img_bgr = _b64_to_bgr(image_b64)

        # 1) Validación cédula (0..1) -> %
        is_valid_id, id_score = self.id_classifier.is_valid_ec_id(id_img_bgr)
        id_score_pct = float(id_score) * 100.0

        # 2) Rostro en cédula (tu FaceDetector retorna dict con "bbox": (x1,y1,x2,y2))
        face_info = self.face_detector.detect(id_img_bgr)
        if not face_info or not face_info.get("bbox"):
            evaluacion = (id_score_pct + 0.0) / 2.0
            return EcuadorIdVerificationResponse(
                status=False,
                message="Cédula válida" if is_valid_id else "La imagen NO corresponde a una cédula válida.",
                payload={
                    "uuidProceso": uuid_proceso,
                    "data": [{"uuid_proceso_cedula": uuid_proceso_cedula, "evaluacion": round(evaluacion, 2)}],
                },
            )

        x1, y1, x2, y2 = face_info["bbox"]
        face_id_bgr = id_img_bgr[y1:y2, x1:x2]

        # 3) Referencia
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
            )

        ref_face_info = self.face_detector.detect(ref_img)
        if not ref_face_info or not ref_face_info.get("bbox"):
            evaluacion = (id_score_pct + 0.0) / 2.0
            return EcuadorIdVerificationResponse(
                status=False,
                message="Cédula válida, pero no se detectó un rostro en la imagen de referencia.",
                payload={
                    "uuidProceso": uuid_proceso,
                    "data": [{"uuid_proceso_cedula": uuid_proceso_cedula, "evaluacion": round(evaluacion, 2)}],
                },
            )

        rx1, ry1, rx2, ry2 = ref_face_info["bbox"]
        face_ref_bgr = ref_img[ry1:ry2, rx1:rx2]

        # 4) Similaridad (0..100) usando tu SimilarityMatcher
        similarity_pct = float(self.similarity.compare(face_id_bgr, face_ref_bgr))

        # 5) Evaluación (promedio)
        evaluacion = (id_score_pct + similarity_pct) / 2.0
        passed = evaluacion > EVALUATION_THRESHOLD
        message = (
            "Cédula válida y rostro coincide con la imagen de referencia."
            if passed else
            "Cédula válida pero el rostro NO coincide con la imagen de referencia."
        )
        if not is_valid_id:
            message = "La imagen NO corresponde a una cédula válida."

        return EcuadorIdVerificationResponse(
            status=passed,
            message=message,
            payload={
                "uuidProceso": uuid_proceso,
                "data": [{"uuid_proceso_cedula": uuid_proceso_cedula, "evaluacion": round(evaluacion, 2)}],
            },
        )
