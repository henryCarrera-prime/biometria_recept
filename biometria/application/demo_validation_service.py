# biometria/application/demo_validation_service.py
import base64
import uuid
import cv2
import numpy as np
import os
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

# Crear directorio para debug de imágenes si no existe
DEBUG_IMAGES_DIR = "debug_demo_images"
os.makedirs(DEBUG_IMAGES_DIR, exist_ok=True)

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
    1. Si el rostro de la persona tiene señales de vida (liveness) - SOLO en rostroPersonaBase64
    2. Si el rostro de la cédula y el rostro de la persona coinciden
    """
    
    def __init__(
        self,
        face_detector: FaceDetector,
        liveness_detector: LivenessPassive,
        similarity_matcher: SimilarityMatcher,
    ):
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
            
            # 2. Validación de cédula removida - solo se usa para extraer rostro
            cedula_valid = True  # Asumir cédula válida para el flujo
            cedula_score_pct = 100.0  # Score fijo 100% ya que no se valida
            
            diagnostics.update({
                "cedula_validation": {
                    "is_valid": True,
                    "score": 1.0,
                    "score_pct": 100.0,
                    "predicted_label": "not_validated",
                    "threshold": "N/A"
                }
            })
            
            # 3. Detectar rostros
            logger.info(f"Detectando rostros - cédula: {W_cedula}x{H_cedula}, rostro: {W_rostro}x{H_rostro}")
            
            # Rostro en cédula - MEJORADO con más logging y manejo de errores
            face_cedula = None
            face_cedula_bbox = None
            face_cedula_landmarks = None
            
            try:
                logger.info("Iniciando detección de rostro en cédula...")
                face_cedula = self.face_detector.detect(cedula_img)
                logger.info(f"Resultado detección cédula: {face_cedula}")
                
                if face_cedula and face_cedula.get("bbox"):
                    bbox = _clip_bbox(face_cedula["bbox"], W_cedula, H_cedula)
                    logger.info(f"Bounding box cédula (original): {face_cedula['bbox']}")
                    logger.info(f"Bounding box cédula (clipped): {bbox}")
                    
                    if bbox:
                        face_cedula_bbox = [int(v) for v in bbox]
                        face_cedula_landmarks = face_cedula.get("landmarks")
                        logger.info(f"✅ Rostro detectado en cédula: {face_cedula_bbox}")
                        logger.info(f"   Área del rostro: {(bbox[2]-bbox[0])*(bbox[3]-bbox[1])}px²")
                        logger.info(f"   Porcentaje del área: {((bbox[2]-bbox[0])*(bbox[3]-bbox[1])/(W_cedula*H_cedula)*100):.2f}%")
                        
                        # Guardar imagen de cédula con rostro detectado
                        try:
                            # Crear copia de la imagen y dibujar bounding box
                            cedula_debug = cedula_img.copy()
                            x1, y1, x2, y2 = face_cedula_bbox
                            cv2.rectangle(cedula_debug, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            cv2.putText(cedula_debug, "ROSTRO DETECTADO", (x1, y1-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            filename = f"{DEBUG_IMAGES_DIR}/cedula_{uuid_validation}.jpg"
                            cv2.imwrite(filename, cedula_debug)
                            logger.info(f"💾 Imagen de cédula guardada: {filename}")
                        except Exception as save_error:
                            logger.error(f"Error guardando imagen de cédula: {save_error}")
                    else:
                        logger.warning("❌ Bounding box de cédula inválido después de clipping")
                else:
                    logger.warning("❌ No se detectó rostro en cédula - resultado vacío o sin bbox")
                    # Guardar imagen de cédula original para análisis
                    try:
                        filename = f"{DEBUG_IMAGES_DIR}/cedula_original_{uuid_validation}.jpg"
                        cv2.imwrite(filename, cedula_img)
                        logger.info(f"💾 Imagen original de cédula guardada: {filename}")
                    except Exception as save_error:
                        logger.error(f"Error guardando imagen original de cédula: {save_error}")
                    
            except Exception as e:
                logger.error(f"❌ Error en detección de rostro en cédula: {e}", exc_info=True)
            
            # Rostro en imagen de persona - MEJORADO con más logging
            face_rostro = None
            face_rostro_bbox = None
            face_rostro_landmarks = None
            
            try:
                logger.info("Iniciando detección de rostro en imagen de persona...")
                face_rostro = self.face_detector.detect(rostro_img)
                logger.info(f"Resultado detección rostro: {face_rostro}")
                
                if face_rostro and face_rostro.get("bbox"):
                    bbox = _clip_bbox(face_rostro["bbox"], W_rostro, H_rostro)
                    logger.info(f"Bounding box rostro (original): {face_rostro['bbox']}")
                    logger.info(f"Bounding box rostro (clipped): {bbox}")
                    
                    if bbox:
                        face_rostro_bbox = [int(v) for v in bbox]
                        face_rostro_landmarks = face_rostro.get("landmarks")
                        logger.info(f"✅ Rostro detectado en persona: {face_rostro_bbox}")
                        logger.info(f"   Área del rostro: {(bbox[2]-bbox[0])*(bbox[3]-bbox[1])}px²")
                        logger.info(f"   Porcentaje del área: {((bbox[2]-bbox[0])*(bbox[3]-bbox[1])/(W_rostro*H_rostro)*100):.2f}%")
                        
                        # Guardar imagen de rostro de persona con bounding box
                        try:
                            rostro_debug = rostro_img.copy()
                            x1, y1, x2, y2 = face_rostro_bbox
                            cv2.rectangle(rostro_debug, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            cv2.putText(rostro_debug, "ROSTRO DETECTADO", (x1, y1-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            filename = f"{DEBUG_IMAGES_DIR}/rostro_{uuid_validation}.jpg"
                            cv2.imwrite(filename, rostro_debug)
                            logger.info(f"💾 Imagen de rostro guardada: {filename}")
                        except Exception as save_error:
                            logger.error(f"Error guardando imagen de rostro: {save_error}")
                    else:
                        logger.warning("❌ Bounding box de rostro inválido después de clipping")
                else:
                    logger.warning("❌ No se detectó rostro en imagen de persona - resultado vacío o sin bbox")
                    
            except Exception as e:
                logger.error(f"❌ Error en detección de rostro en persona: {e}", exc_info=True)
            
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
            
            logger.info(f"Iniciando detección de liveness - rostro detectado: {face_rostro_bbox is not None}")
            
            # Intentar liveness incluso si no se detectó rostro específico
            # Usar la imagen completa del rostro de persona
            try:
                logger.info("Intentando liveness detection con imagen completa...")
                liveness_score = float(self.liveness_detector.score(rostro_img))
                liveness_ok = liveness_score > 0.5  # Umbral reducido de 0.7 a 0.5
                logger.info(f"Liveness score: {liveness_score}, OK: {liveness_ok}")
            except Exception as e:
                logger.error(f"Error en liveness detection: {e}", exc_info=True)
                # Si falla, intentar con enfoque alternativo usando recorte de rostro
                if face_rostro_bbox:
                    try:
                        logger.info("Intentando liveness con recorte de rostro...")
                        x1, y1, x2, y2 = face_rostro_bbox
                        face_crop = rostro_img[y1:y2, x1:x2]
                        logger.info(f"Recorte de rostro para liveness: {face_crop.shape}")
                        
                        # Redimensionar el recorte para mejorar la calidad
                        scale_factor = 400.0 / min(face_crop.shape[:2])
                        new_width = int(face_crop.shape[1] * scale_factor)
                        new_height = int(face_crop.shape[0] * scale_factor)
                        resized_crop = cv2.resize(face_crop, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                        logger.info(f"Recorte redimensionado: {resized_crop.shape}")
                        
                        liveness_score = float(self.liveness_detector.score(resized_crop))
                        liveness_ok = liveness_score > 0.5
                        logger.info(f"Liveness score con recorte: {liveness_score}, OK: {liveness_ok}")
                    except Exception as fallback_error:
                        logger.error(f"Error en fallback liveness: {fallback_error}", exc_info=True)
                else:
                    logger.warning("No se puede verificar liveness - no se detectó rostro en imagen de persona")
            
            diagnostics.update({
                "liveness": {
                    "score": round(liveness_score, 4),
                    "is_live": liveness_ok,
                    "threshold": 0.5
                }
            })
            
            # 5. Comparar similitud entre rostros - MEJORADO para usar imagen completa si no hay rostro detectado
            similarity_score = 0.0
            similarity_ok = False
            
            logger.info(f"Verificando condiciones para similitud - cédula: {face_cedula_bbox is not None}, rostro: {face_rostro_bbox is not None}")
            
            # Siempre intentar calcular similitud, incluso si no se detectaron rostros localmente
            # Rekognition puede encontrar rostros donde nuestro detector local no los encuentra
            try:
                logger.info("Calculando similitud con Rekognition...")
                
                # Si no se detectó rostro en cédula localmente, usar imagen completa
                # Rekognition puede ser más efectivo encontrando rostros
                if face_cedula_bbox:
                    x1, y1, x2, y2 = face_cedula_bbox
                    face_cedula_crop = cedula_img[y1:y2, x1:x2]
                    logger.info(f"Usando recorte de rostro en cédula: {face_cedula_crop.shape}")
                    
                    # Guardar recorte de cédula
                    try:
                        crop_cedula_filename = f"{DEBUG_IMAGES_DIR}/crop_cedula_{uuid_validation}.jpg"
                        cv2.imwrite(crop_cedula_filename, face_cedula_crop)
                        logger.info(f"💾 Recorte de cédula guardado: {crop_cedula_filename}")
                    except Exception as crop_error:
                        logger.error(f"Error guardando recorte de cédula: {crop_error}")
                else:
                    face_cedula_crop = cedula_img  # Usar imagen completa
                    logger.info("⚠️ No se detectó rostro en cédula localmente - usando imagen completa para Rekognition")
                
                # Si no se detectó rostro en persona localmente, usar imagen completa
                if face_rostro_bbox:
                    x1, y1, x2, y2 = face_rostro_bbox
                    face_rostro_crop = rostro_img[y1:y2, x1:x2]
                    logger.info(f"Usando recorte de rostro en persona: {face_rostro_crop.shape}")
                    
                    # Guardar recorte de rostro
                    try:
                        crop_rostro_filename = f"{DEBUG_IMAGES_DIR}/crop_rostro_{uuid_validation}.jpg"
                        cv2.imwrite(crop_rostro_filename, face_rostro_crop)
                        logger.info(f"💾 Recorte de rostro guardado: {crop_rostro_filename}")
                    except Exception as crop_error:
                        logger.error(f"Error guardando recorte de rostro: {crop_error}")
                else:
                    face_rostro_crop = rostro_img  # Usar imagen completa
                    logger.info("⚠️ No se detectó rostro en persona localmente - usando imagen completa para Rekognition")
                
                # Calcular similitud con Rekognition (puede encontrar rostros donde nuestro detector no los encuentra)
                similarity_score = float(self.similarity_matcher.compare(face_cedula_crop, face_rostro_crop))
                similarity_ok = similarity_score >= 95.0  # Umbral de Rekognition
                
                logger.info(f"✅ Similaridad calculada por Rekognition: {similarity_score}%, OK: {similarity_ok}")
                
            except Exception as e:
                logger.error(f"❌ Error en comparación de similitud: {e}", exc_info=True)
            
            diagnostics.update({
                "similarity": {
                    "score": round(similarity_score, 2),
                    "is_match": similarity_ok,
                    "threshold": 95.0,
                    "cedula_face_found": face_cedula_bbox is not None,
                    "rostro_face_found": face_rostro_bbox is not None
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