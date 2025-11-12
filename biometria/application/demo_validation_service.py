# biometria/application/demo_validation_service.py
import base64
import uuid
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import logging

from biometria.domain.interfaces import (
    FaceDetector,
    SimilarityMatcher,
    LivenessPassive,
)
from biometria.infrastructure.classifier.keras_cedula_classifier import KerasEcuadorIdClassifier

logger = logging.getLogger("biometria.verify")

def _b64_to_bgr(b64: str) -> np.ndarray:
    """Convierte base64 a imagen BGR (OpenCV)"""
    data = base64.b64decode(b64.split(",")[-1])
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("imageBase64 inválido (no se pudo decodificar)")
    return img

def _clip_bbox(bbox, w, h):
    """Ajusta bounding box a los límites de la imagen"""
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)

@dataclass
class DemoValidationResponse:
    """Respuesta del servicio de validación demo"""
    status: bool
    message: str
    payload: Dict[str, Any]
    diagnostics: Optional[Dict[str, Any]] = None

class DemoValidationService:
    """
    Servicio que valida:
    1. Si la imagen de la cédula es una cédula válida
    2. Si el rostro de la persona tiene señales de vida (liveness) - SOLO en rostroPersonaBase64
    3. Si el rostro de la cédula y el rostro de la persona coinciden
    """
    
    def __init__(
        self,
        id_classifier: KerasEcuadorIdClassifier,
        face_detector: FaceDetector,
        liveness_detector: LivenessPassive,
        similarity_matcher: SimilarityMatcher,
    ):
        self.id_classifier = id_classifier
        self.face_detector = face_detector
        self.liveness_detector = liveness_detector
        self.similarity_matcher = similarity_matcher

    def execute(
        self, 
        uuid_proceso: str, 
        cedula_frontal_b64: str, 
        rostro_persona_b64: str
    ) -> DemoValidationResponse:
        """
        Ejecuta la validación completa
        
        Args:
            uuid_proceso: UUID del proceso
            cedula_frontal_b64: Imagen de la cédula frontal en base64
            rostro_persona_b64: Imagen del rostro de la persona en base64
            
        Returns:
            DemoValidationResponse con resultados completos
        """
        uuid_validation = str(uuid.uuid4())
        
        try:
            # 1. Decodificar imágenes
            cedula_img = _b64_to_bgr(cedula_frontal_b64)
            rostro_img = _b64_to_bgr(rostro_persona_b64)
            
            H_cedula, W_cedula = cedula_img.shape[:2]
            H_rostro, W_rostro = rostro_img.shape[:2]
            
            diagnostics = {
                "cedula_image_size": f"{W_cedula}x{H_cedula}",
                "rostro_image_size": f"{W_rostro}x{H_rostro}",
                "uuid_validation": uuid_validation
            }
            
            # 2. Validar cédula
            cedula_valid, cedula_score = self.id_classifier.is_valid_ec_id(cedula_img)
            cedula_score_pct = float(cedula_score) * 100.0
            
            # Obtener label predicho
            predicted_label = self.id_classifier.predict_label(cedula_img)
            
            diagnostics.update({
                "cedula_validation": {
                    "is_valid": bool(cedula_valid),
                    "score": round(cedula_score, 4),
                    "score_pct": round(cedula_score_pct, 2),
                    "predicted_label": predicted_label,
                    "threshold": self.id_classifier.th
                }
            })
            
            # 3. Detectar rostros
            # Rostro en cédula
            face_cedula = self.face_detector.detect(cedula_img)
            face_cedula_bbox = None
            face_cedula_landmarks = None
            
            if face_cedula and face_cedula.get("bbox"):
                bbox = _clip_bbox(face_cedula["bbox"], W_cedula, H_cedula)
                if bbox:
                    face_cedula_bbox = [int(v) for v in bbox]
                    face_cedula_landmarks = face_cedula.get("landmarks")
            
            # Rostro en imagen de persona
            face_rostro = self.face_detector.detect(rostro_img)
            face_rostro_bbox = None
            face_rostro_landmarks = None
            
            if face_rostro and face_rostro.get("bbox"):
                bbox = _clip_bbox(face_rostro["bbox"], W_rostro, H_rostro)
                if bbox:
                    face_rostro_bbox = [int(v) for v in bbox]
                    face_rostro_landmarks = face_rostro.get("landmarks")
            
            diagnostics.update({
                "face_detection": {
                    "cedula": {
                        "face_found": face_cedula_bbox is not None,
                        "bbox": face_cedula_bbox,
                        "landmarks_count": len(face_cedula_landmarks) if face_cedula_landmarks else 0
                    },
                    "rostro": {
                        "face_found": face_rostro_bbox is not None,
                        "bbox": face_rostro_bbox,
                        "landmarks_count": len(face_rostro_landmarks) if face_rostro_landmarks else 0
                    }
                }
            })
            
            # 4. Verificar liveness SOLO en rostro de persona (no en cédula)
            liveness_score = 0.0
            liveness_ok = False
            
            if face_rostro_bbox:
                try:
                    # Verificar el tamaño de la imagen y usar la mejor opción para liveness
                    H_rostro, W_rostro = rostro_img.shape[:2]
                    
                    # Si la imagen completa es demasiado pequeña (< 400px en lado corto), usar un enfoque alternativo
                    min_dimension = min(H_rostro, W_rostro)
                    if min_dimension < 400:
                        logger.warning(f"Imagen de rostro muy pequeña: {W_rostro}x{H_rostro}. Intentando mejorar calidad...")
                        
                        # Intentar usar el recorte del rostro pero redimensionado
                        x1, y1, x2, y2 = face_rostro_bbox
                        face_crop = rostro_img[y1:y2, x1:x2]
                        
                        # Redimensionar el recorte para mejorar la calidad
                        scale_factor = 400.0 / min(face_crop.shape[:2])
                        new_width = int(face_crop.shape[1] * scale_factor)
                        new_height = int(face_crop.shape[0] * scale_factor)
                        resized_crop = cv2.resize(face_crop, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                        
                        liveness_score = float(self.liveness_detector.score(resized_crop))
                    else:
                        # Usar imagen completa si tiene suficiente tamaño
                        liveness_score = float(self.liveness_detector.score(rostro_img))
                    
                    liveness_ok = liveness_score > 0.7  # Umbral ajustable
                    logger.info(f"Liveness score obtenido: {liveness_score}, OK: {liveness_ok}")
                    
                except Exception as e:
                    logger.error(f"Error en liveness detection: {e}")
                    # En caso de error, intentar con la imagen completa como fallback
                    try:
                        liveness_score = float(self.liveness_detector.score(rostro_img))
                        liveness_ok = liveness_score > 0.7
                        logger.info(f"Liveness fallback score: {liveness_score}")
                    except Exception as fallback_error:
                        logger.error(f"Error en fallback liveness: {fallback_error}")
            
            diagnostics.update({
                "liveness": {
                    "score": round(liveness_score, 4),
                    "is_live": liveness_ok,
                    "threshold": 0.7
                }
            })
            
            # 5. Comparar similitud entre rostros
            similarity_score = 0.0
            similarity_ok = False
            
            if face_cedula_bbox and face_rostro_bbox:
                try:
                    # Extraer rostros de las imágenes completas
                    if face_cedula_bbox:
                        x1, y1, x2, y2 = face_cedula_bbox
                        face_cedula_crop = cedula_img[y1:y2, x1:x2]
                    else:
                        face_cedula_crop = cedula_img
                    
                    if face_rostro_bbox:
                        x1, y1, x2, y2 = face_rostro_bbox
                        face_rostro_crop = rostro_img[y1:y2, x1:x2]
                    else:
                        face_rostro_crop = rostro_img
                    
                    similarity_score = float(self.similarity_matcher.compare(face_cedula_crop, face_rostro_crop))
                    similarity_ok = similarity_score >= 95.0  # Umbral de Rekognition
                    
                except Exception as e:
                    logger.error(f"Error en comparación de similitud: {e}")
            
            diagnostics.update({
                "similarity": {
                    "score": round(similarity_score, 2),
                    "is_match": similarity_ok,
                    "threshold": 95.0
                }
            })
            
            # 6. Evaluación final
            all_checks_passed = (
                cedula_valid and 
                face_cedula_bbox is not None and 
                face_rostro_bbox is not None and
                liveness_ok and 
                similarity_ok
            )
            
            evaluation_pct = round((
                cedula_score_pct + 
                (liveness_score * 100) + 
                similarity_score
            ) / 3.0, 2)
            
            # Construir mensaje descriptivo
            if all_checks_passed:
                message = "Validación exitosa: cédula válida, liveness detectado y rostros coinciden."
            else:
                failures = []
                if not cedula_valid:
                    failures.append("cédula no válida")
                if not face_cedula_bbox:
                    failures.append("no se detectó rostro en cédula")
                if not face_rostro_bbox:
                    failures.append("no se detectó rostro en imagen de persona")
                if not liveness_ok:
                    failures.append("liveness no detectado")
                if not similarity_ok:
                    failures.append("rostros no coinciden")
                
                message = f"Validación fallida: {', '.join(failures)}."
            
            payload = {
                "uuidProceso": uuid_proceso,
                "data": [{
                    "uuid_validation": uuid_validation,
                    "evaluacion": evaluation_pct,
                    "cedula_valida": bool(cedula_valid),
                    "liveness_detectado": liveness_ok,
                    "rostros_coinciden": similarity_ok,
                    "score_cedula": round(cedula_score_pct, 2),
                    "score_liveness": round(liveness_score * 100, 2),
                    "score_similarity": round(similarity_score, 2)
                }]
            }
            
            return DemoValidationResponse(
                status=all_checks_passed,
                message=message,
                payload=payload,
                diagnostics=diagnostics
            )
            
        except Exception as e:
            logger.error(f"Error en demo_validation_service: {e}")
            return DemoValidationResponse(
                status=False,
                message=f"Error procesando validación: {str(e)}",
                payload={
                    "uuidProceso": uuid_proceso,
                    "data": [{"uuid_validation": uuid_validation, "evaluacion": 0.0}]
                },
                diagnostics={"error": str(e)}
            )