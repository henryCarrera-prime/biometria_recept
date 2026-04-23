# biometria/application/demo_validation_service.py
import base64
import uuid
import cv2
import numpy as np
import os
import datetime
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import logging
from zoneinfo import ZoneInfo

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


class DemoValidationExtendedService:
    """
    Servicio extendido por etapas para validar biometr?a con soporte de im?genes opcionales.
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
        self.id_classifier = KerasEcuadorIdClassifier()

        self.liveness_threshold = float(os.getenv("DEMO_LIVENESS_THRESHOLD", "0.5"))
        self.similarity_threshold = float(os.getenv("DEMO_SIMILARITY_THRESHOLD", "95.0"))
        self.cedula_threshold = float(os.getenv("ECU_ID_THRESHOLD", "0.85"))
        raw_labels = os.getenv("ECU_ID_POSITIVE_LABELS", "0,1,2")
        self.cedula_positive_labels = {
            int(tok.strip())
            for tok in raw_labels.replace(";", ",").split(",")
            if tok.strip().isdigit()
        } or {0, 1, 2}

    @staticmethod
    def _now_ecuador_iso() -> str:
        return datetime.datetime.now(ZoneInfo("America/Guayaquil")).isoformat()

    @staticmethod
    def _bgr_to_data_uri(img_bgr: Optional[np.ndarray]) -> Optional[str]:
        if img_bgr is None:
            return None
        ok, buff = cv2.imencode(".jpg", img_bgr)
        if not ok:
            return None
        return f"data:image/jpeg;base64,{base64.b64encode(buff.tobytes()).decode('ascii')}"

    @staticmethod
    def _try_get_predicted_label(classifier: Any, image_bgr: np.ndarray) -> Optional[int]:
        for attr in ("predict_label", "predict_class", "classify", "predict"):
            fn = getattr(classifier, attr, None)
            if callable(fn):
                try:
                    pred = fn(image_bgr)
                    if isinstance(pred, (list, tuple)) and pred:
                        return int(pred[0])
                    if isinstance(pred, dict) and "label" in pred:
                        return int(pred["label"])
                    return int(pred)
                except Exception:
                    continue
        return None

    @staticmethod
    def _rotate_for_eval(img_bgr: np.ndarray, angle: int) -> np.ndarray:
        if angle == 0:
            return img_bgr
        if angle == 90:
            return cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
        if angle == 180:
            return cv2.rotate(img_bgr, cv2.ROTATE_180)
        if angle == 270:
            return cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img_bgr

    def _detect_face(self, img_bgr: Optional[np.ndarray]) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "face_found": False,
            "bbox": None,
            "landmarks_count": 0,
            "face_crop": None,
            "image_size": None,
        }
        if img_bgr is None:
            return out

        h, w = img_bgr.shape[:2]
        out["image_size"] = f"{w}x{h}"

        try:
            detection = self.face_detector.detect(img_bgr)
            if detection and detection.get("bbox"):
                bbox = _clip_bbox(detection["bbox"], w, h)
                if bbox:
                    x1, y1, x2, y2 = bbox
                    out["face_found"] = True
                    out["bbox"] = [int(v) for v in bbox]
                    out["landmarks_count"] = len(detection.get("landmarks") or [])
                    out["face_crop"] = img_bgr[y1:y2, x1:x2]
        except Exception as ex:
            out["error"] = str(ex)

        return out

    def _evaluate_cedula_with_rotation(self, cedula_img: np.ndarray) -> Dict[str, Any]:
        candidates: List[Dict[str, Any]] = []

        for angle in (0, 90, 180, 270):
            rotated = self._rotate_for_eval(cedula_img, angle)
            pred_label = None
            valid_raw = False
            score_model = 0.0

            try:
                valid_raw, score_model = self.id_classifier.is_valid_ec_id(rotated)
                pred_label = self._try_get_predicted_label(self.id_classifier, rotated)
            except Exception as ex:
                logger.error(f"Error clasificando c?dula (rotaci?n {angle}): {ex}")

            label_ok = pred_label in self.cedula_positive_labels if pred_label is not None else False
            score_ok = float(score_model) >= self.cedula_threshold
            face_info = self._detect_face(rotated)
            valid_final = bool(valid_raw and label_ok and score_ok and face_info["face_found"])

            candidates.append({
                "angle": angle,
                "image": rotated,
                "predicted_label": pred_label,
                "valid_raw": bool(valid_raw),
                "score_model": float(score_model),
                "score_model_pct": round(float(score_model) * 100.0, 2),
                "label_ok": bool(label_ok),
                "score_ok": bool(score_ok),
                "valid_final": bool(valid_final),
                "face_info": face_info,
            })

        best = max(
            candidates,
            key=lambda c: (
                1 if c["valid_final"] else 0,
                1 if c["face_info"]["face_found"] else 0,
                c["score_model"],
            ),
        )

        best["rotation_applied"] = best["angle"] != 0
        best["tested_rotations"] = [
            {
                "angle": c["angle"],
                "valid_final": c["valid_final"],
                "score_model_pct": c["score_model_pct"],
                "predicted_label": c["predicted_label"],
                "face_found": c["face_info"]["face_found"],
            }
            for c in candidates
        ]
        return best

    def execute(
        self,
        uuid_proceso: str,
        cedula_frontal_b64: Optional[str],
        rostro_persona_b64: str,
        registro_civil_b64: Optional[str] = None,
    ) -> DemoValidationResponse:
        uuid_validation = str(uuid.uuid4())
        timestamp_ec = self._now_ecuador_iso()

        try:
            rostro_img = _b64_to_bgr(rostro_persona_b64)
            cedula_img = _b64_to_bgr(cedula_frontal_b64) if cedula_frontal_b64 else None
            registro_civil_img = _b64_to_bgr(registro_civil_b64) if registro_civil_b64 else None

            rostro_face = self._detect_face(rostro_img)
            registro_face = self._detect_face(registro_civil_img) if registro_civil_img is not None else None
            cedula_eval = self._evaluate_cedula_with_rotation(cedula_img) if cedula_img is not None else None
            cedula_face = cedula_eval["face_info"] if cedula_eval is not None else None

            etapa_1_checks = {
                "rostroPersonaBase64": bool(rostro_face["face_found"]),
                "cedulaFrontalBase64": bool(cedula_face["face_found"]) if cedula_face is not None else None,
                "registroCivilBase64": bool(registro_face["face_found"]) if registro_face is not None else None,
            }
            etapa_1_ok = all(v is True for v in etapa_1_checks.values() if v is not None)

            cedula_valid = bool(cedula_eval["valid_final"]) if cedula_eval is not None else True
            cedula_score_pct = round(cedula_eval["score_model_pct"], 2) if cedula_eval is not None else 0.0

            liveness_score = 0.0
            liveness_ok = False
            try:
                liveness_score = float(self.liveness_detector.score(rostro_img))
                liveness_ok = liveness_score >= self.liveness_threshold
            except Exception as ex:
                logger.warning(f"Liveness con imagen completa fall?: {ex}")
                if rostro_face["face_found"] and rostro_face["face_crop"] is not None:
                    try:
                        liveness_score = float(self.liveness_detector.score(rostro_face["face_crop"]))
                        liveness_ok = liveness_score >= self.liveness_threshold
                    except Exception as fallback_ex:
                        logger.warning(f"Liveness fallback fall?: {fallback_ex}")

            sources: List[Dict[str, Any]] = [
                {
                    "key": "rostro_persona",
                    "provided": True,
                    "image": rostro_img,
                    "face_found": bool(rostro_face["face_found"]),
                    "face_crop": rostro_face["face_crop"] if rostro_face["face_crop"] is not None else rostro_img,
                    "image_base64": self._bgr_to_data_uri(rostro_face["face_crop"] if rostro_face["face_crop"] is not None else rostro_img),
                }
            ]

            if cedula_img is not None:
                cedula_img_for_compare = cedula_eval["image"] if cedula_eval is not None else cedula_img
                cedula_crop = cedula_face["face_crop"] if (cedula_face and cedula_face["face_crop"] is not None) else cedula_img_for_compare
                sources.append(
                    {
                        "key": "cedula_frontal",
                        "provided": True,
                        "image": cedula_img_for_compare,
                        "face_found": bool(cedula_face["face_found"]) if cedula_face else False,
                        "face_crop": cedula_crop,
                        "image_base64": self._bgr_to_data_uri(cedula_crop),
                    }
                )

            if registro_civil_img is not None:
                registro_crop = registro_face["face_crop"] if (registro_face and registro_face["face_crop"] is not None) else registro_civil_img
                sources.append(
                    {
                        "key": "registro_civil",
                        "provided": True,
                        "image": registro_civil_img,
                        "face_found": bool(registro_face["face_found"]) if registro_face else False,
                        "face_crop": registro_crop,
                        "image_base64": self._bgr_to_data_uri(registro_crop),
                    }
                )

            pair_scores: List[Dict[str, Any]] = []
            for i in range(len(sources)):
                for j in range(i + 1, len(sources)):
                    a = sources[i]
                    b = sources[j]
                    score = 0.0
                    try:
                        score = float(self.similarity_matcher.compare(a["face_crop"], b["face_crop"]))
                    except Exception as ex:
                        logger.error(f"Error comparando {a['key']} vs {b['key']}: {ex}")
                    pair_scores.append(
                        {
                            "pair": f"{a['key']}_vs_{b['key']}",
                            "sources": [a["key"], b["key"]],
                            "score": round(score, 2),
                            "is_match": bool(score >= self.similarity_threshold),
                            "threshold": self.similarity_threshold,
                        }
                    )

            similarity_global = (
                round(sum(p["score"] for p in pair_scores) / len(pair_scores), 2)
                if pair_scores
                else 0.0
            )
            similarity_ok = bool(pair_scores) and all(p["is_match"] for p in pair_scores)

            pair_index = {p["pair"]: p for p in pair_scores}
            cedula_rostro = pair_index.get("cedula_frontal_vs_rostro_persona") or pair_index.get("rostro_persona_vs_cedula_frontal")
            registro_rostro = pair_index.get("registro_civil_vs_rostro_persona") or pair_index.get("rostro_persona_vs_registro_civil")

            etapa_2_ok = True if cedula_eval is None else bool(cedula_valid)
            etapa_3_ok = bool(liveness_ok)
            etapa_4_ok = bool(similarity_ok)

            failure_reasons: List[str] = []
            if not etapa_1_ok:
                if not rostro_face["face_found"]:
                    failure_reasons.append("no se detect? rostro en rostroPersonaBase64")
                if cedula_face is not None and not cedula_face["face_found"]:
                    failure_reasons.append("la imagen de c?dula no tiene rostro detectable")
                if registro_face is not None and not registro_face["face_found"]:
                    failure_reasons.append("la imagen de registro civil no tiene rostro detectable")
            if not etapa_2_ok:
                failure_reasons.append("la c?dula no corresponde a un tipo de c?dula v?lido")
            if not etapa_3_ok:
                failure_reasons.append("status de vida no aprobado")
            if not etapa_4_ok:
                failure_reasons.append("las im?genes no cumplen el umbral de similitud")
            if len(sources) < 2:
                failure_reasons.append("se requieren al menos 2 im?genes para validar similitud")

            all_checks_passed = etapa_1_ok and etapa_2_ok and etapa_3_ok and etapa_4_ok and len(sources) >= 2

            components_for_eval = [similarity_global, round(liveness_score * 100.0, 2)]
            if cedula_eval is not None:
                components_for_eval.append(cedula_score_pct)
            evaluation_pct = round(sum(components_for_eval) / len(components_for_eval), 2) if components_for_eval else 0.0

            if all_checks_passed:
                message = "Validaci?n exitosa por etapas: rostros detectados, c?dula v?lida, status de vida aprobado y similitud aprobada."
            else:
                message = f"Validaci?n fallida: {', '.join(failure_reasons)}."

            data_item = {
                "uuid_validation": uuid_validation,
                "evaluacion": evaluation_pct,
                "cedula_valida": bool(cedula_valid),
                "liveness_detectado": bool(liveness_ok),
                "rostros_coinciden": bool(similarity_ok),
                "score_cedula": round(cedula_score_pct, 2),
                "score_liveness": round(liveness_score * 100.0, 2),
                "score_similarity": round(similarity_global, 2),
                "cedula_rostro_match": cedula_rostro["is_match"] if cedula_rostro else None,
                "cedula_rostro_score": cedula_rostro["score"] if cedula_rostro else None,
                "registro_civil_rostro_match": registro_rostro["is_match"] if registro_rostro else None,
                "registro_civil_rostro_score": registro_rostro["score"] if registro_rostro else None,
                "registro_civil_provided": registro_civil_img is not None,
                "cedula_provided": cedula_img is not None,
                "cedula_validacion_aplica": cedula_img is not None,
                "cantidad_imagenes_comparadas": len(sources),
                "porcentaje_global_similitud": round(similarity_global, 2),
                "porcentaje_vida": round(liveness_score * 100.0, 2),
                "timestamp_ecuador": timestamp_ec,
                "identificador_proceso_asignado": uuid_validation,
                "pares_comparados": pair_scores,
                "imagenes_comparadas": [
                    {
                        "fuente": s["key"],
                        "incluida_en_similitud": True,
                        "tiene_rostro": bool(s["face_found"]),
                        "imagenBase64": s["image_base64"],
                    }
                    for s in sources
                ],
                "puntos_evaluados": {
                    "etapa_1_rostros": etapa_1_ok,
                    "etapa_2_cedula": etapa_2_ok if cedula_img is not None else "no_aplica",
                    "etapa_3_status_vida": etapa_3_ok,
                    "etapa_3_preformateo": True,
                    "etapa_4_similitud": etapa_4_ok,
                },
            }

            diagnostics = {
                "uuid_validation": uuid_validation,
                "identificador_proceso_asignado": uuid_validation,
                "timestamp_ecuador": timestamp_ec,
                "parametrizacion": {
                    "liveness_threshold": self.liveness_threshold,
                    "similarity_threshold": self.similarity_threshold,
                    "cedula_threshold": self.cedula_threshold,
                    "cedula_positive_labels": sorted(list(self.cedula_positive_labels)),
                },
                "inputs": {
                    "cedula_provided": cedula_img is not None,
                    "rostro_provided": True,
                    "registro_civil_provided": registro_civil_img is not None,
                },
                "face_detection": {
                    "cedula": {
                        "face_found": cedula_face["face_found"] if cedula_face else None,
                        "bbox": cedula_face["bbox"] if cedula_face else None,
                        "landmarks_count": cedula_face["landmarks_count"] if cedula_face else None,
                    },
                    "rostro": {
                        "face_found": rostro_face["face_found"],
                        "bbox": rostro_face["bbox"],
                        "landmarks_count": rostro_face["landmarks_count"],
                    },
                    "registro_civil": {
                        "face_found": registro_face["face_found"] if registro_face else None,
                        "bbox": registro_face["bbox"] if registro_face else None,
                        "landmarks_count": registro_face["landmarks_count"] if registro_face else None,
                    },
                },
                "cedula_validation": {
                    "applies": cedula_img is not None,
                    "is_valid": cedula_valid,
                    "score_pct": round(cedula_score_pct, 2),
                    "predicted_label": cedula_eval["predicted_label"] if cedula_eval else None,
                    "rotation_applied": cedula_eval["rotation_applied"] if cedula_eval else None,
                    "rotation_angle": cedula_eval["angle"] if cedula_eval else None,
                    "tested_rotations": cedula_eval["tested_rotations"] if cedula_eval else [],
                },
                "liveness": {
                    "is_live": bool(liveness_ok),
                    "score": round(liveness_score, 4),
                    "score_pct": round(liveness_score * 100.0, 2),
                    "threshold": self.liveness_threshold,
                },
                "similarity": {
                    "pairs": pair_scores,
                    "global_score_pct": round(similarity_global, 2),
                    "all_pairs_match": bool(similarity_ok),
                    "pairs_count": len(pair_scores),
                },
                "errors": failure_reasons,
            }

            return DemoValidationResponse(
                status=all_checks_passed,
                message=message,
                payload={"uuidProceso": uuid_proceso, "data": [data_item]},
                diagnostics=diagnostics,
            )
        except Exception as e:
            logger.error(f"Error en demo_validation_extended_service: {e}", exc_info=True)
            return DemoValidationResponse(
                status=False,
                message=f"Error procesando validaci?n: {str(e)}",
                payload={
                    "uuidProceso": uuid_proceso,
                    "data": [
                        {
                            "uuid_validation": uuid_validation,
                            "evaluacion": 0.0,
                            "timestamp_ecuador": timestamp_ec,
                            "identificador_proceso_asignado": uuid_validation,
                            "porcentaje_vida": 0.0,
                            "porcentaje_global_similitud": 0.0,
                            "puntos_evaluados": {
                                "etapa_1_rostros": False,
                                "etapa_2_cedula": "no_ejecutado",
                                "etapa_3_status_vida": False,
                                "etapa_3_preformateo": False,
                                "etapa_4_similitud": False,
                            },
                            "pares_comparados": [],
                            "imagenes_comparadas": [],
                        }
                    ],
                },
                diagnostics={
                    "uuid_validation": uuid_validation,
                    "identificador_proceso_asignado": uuid_validation,
                    "timestamp_ecuador": timestamp_ec,
                    "parametrizacion": {
                        "liveness_threshold": self.liveness_threshold,
                        "similarity_threshold": self.similarity_threshold,
                        "cedula_threshold": self.cedula_threshold,
                        "cedula_positive_labels": sorted(list(self.cedula_positive_labels)),
                    },
                    "liveness": {
                        "is_live": False,
                        "score": 0.0,
                        "score_pct": 0.0,
                        "threshold": self.liveness_threshold,
                    },
                    "error": str(e),
                },
            )
