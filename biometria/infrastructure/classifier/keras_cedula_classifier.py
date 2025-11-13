# biometria/infrastructure/id_classifier/keras_ecuador_id_classifier.py
import os, threading, numpy as np, cv2
import tensorflow as tf
import re
import logging
from typing import Optional, Tuple

logger = logging.getLogger("biometria.verify")
_model = None
_model_lock = threading.Lock()

def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(raw))
    return float(m.group(0)) if m else float(default)

def _parse_size() -> Tuple[int, int]:
    s = os.getenv("ECU_ID_INPUT_SIZE", "224x224").lower().replace(" ", "")
    try:
        w, h = s.split("x")
        return int(w), int(h)
    except Exception:
        return (224, 224)

def _load_tf_model(path: str):
    # 1) Carga "limpia" con tf.keras
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        msg = str(e)
        # 2) Parche puntual para modelos con DepthwiseConv2D(groups=..)
        if "DepthwiseConv2D" in msg and "Unrecognized keyword" in msg and "groups" in msg:
            class DepthwiseConv2DPatched(tf.keras.layers.DepthwiseConv2D):
                @classmethod
                def from_config(cls, cfg):
                    cfg.pop("groups", None)
                    return super().from_config(cfg)
            return tf.keras.models.load_model(
                path, compile=False, custom_objects={"DepthwiseConv2D": DepthwiseConv2DPatched}
            )
        raise

def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                # Usar ruta relativa al directorio del proyecto
                default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "models", "keras_model.h5")
                path = os.getenv("ECU_ID_MODEL_PATH", default_path)
                logger.info(f"Cargando modelo Keras desde: {path}")
                
                # Verificar si el archivo existe
                if not os.path.exists(path):
                    logger.warning(f"Archivo de modelo no encontrado en: {path}")
                    logger.info("Usando modo fallback sin modelo Keras")
                    _model = None
                    return _model
                
                try:
                    _model = _load_tf_model(path)
                    # Log información detallada del modelo cargado
                    logger.info("=== INFORMACIÓN DEL MODELO CARGADO ===")
                    if hasattr(_model, 'inputs'):
                        logger.info(f"Entradas: {len(_model.inputs)}")
                        for i, inp in enumerate(_model.inputs):
                            logger.info(f"  Input {i}: {inp.shape} (dtype: {inp.dtype})")
                    else:
                        logger.warning("Modelo cargado pero no tiene atributo 'inputs'")
                    
                    if hasattr(_model, 'outputs'):
                        logger.info(f"Salidas: {len(_model.outputs)}")
                        for i, out in enumerate(_model.outputs):
                            logger.info(f"  Output {i}: {out.shape} (dtype: {out.dtype})")
                    
                    if hasattr(_model, 'input_shape'):
                        logger.info(f"Input shape: {_model.input_shape}")
                    
                    if hasattr(_model, 'summary'):
                        # Log resumen del modelo (solo estructura básica)
                        logger.info("Resumen del modelo:")
                        _model.summary(print_fn=lambda x: logger.info(f"  {x}"))
                    
                    logger.info("=== FIN INFORMACIÓN DEL MODELO ===")
                except Exception as e:
                    logger.error(f"No se pudo cargar el modelo desde {path}. Error: {e}", exc_info=True)
                    logger.info("Usando modo fallback sin modelo Keras")
                    _model = None
    return _model

class KerasEcuadorIdClassifier:
    """
    - Entrada: imagen BGR (OpenCV)
    - Preproc: BGR->RGB, resize (ECU_ID_INPUT_SIZE o 224x224), normaliza [0,1]
    - Salida: softmax N clases (0..N-1)
    - Expone:
        * is_valid_ec_id(img_bgr) -> (bool, score_top1)
        * predict_label(img_bgr=None) -> int (top-1)
        * last_probs (propiedad) -> np.ndarray | None
    """
    def __init__(self):
        self.size = _parse_size()
        self.th = _env_float("ECU_ID_THRESHOLD", 0.85)
        self._last_probs: Optional[np.ndarray] = None
        self._last_label: Optional[int] = None

    def _prep(self, img_bgr: np.ndarray) -> np.ndarray:
        logger.info(f"Preparando imagen - tamaño original: {img_bgr.shape}")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        logger.info(f"Tamaño objetivo del modelo: {self.size}")
        img = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
        logger.info(f"Imagen redimensionada: {img.shape}")
        img = img.astype("float32") / 255.0
        result = np.expand_dims(img, axis=0)  # (1,H,W,3)
        logger.info(f"Tensor final para modelo: {result.shape}")
        return result

    def _infer(self, image_bgr: np.ndarray) -> np.ndarray:
        logger.info(f"Iniciando inferencia - imagen de entrada: {image_bgr.shape}")
        model = _get_model()
        if model is None:
            # Modo fallback: devolver probabilidades que indican cédula no válida
            logger.warning("Modelo Keras no disponible, usando modo fallback")
            probs = np.array([1.0, 0.0], dtype=np.float32)  # [probabilidad de no válido, probabilidad de válido]
            self._last_probs = probs
            self._last_label = 0
            return probs
        
        try:
            # Preparar la imagen para el modelo
            input_tensor = self._prep(image_bgr)
            logger.info(f"Tensor de entrada para predict: {input_tensor.shape}")
            
            # Verificar las dimensiones esperadas por el modelo
            if hasattr(model, 'input_shape'):
                logger.info(f"Modelo espera input_shape: {model.input_shape}")
            if hasattr(model, 'inputs') and model.inputs:
                for i, inp in enumerate(model.inputs):
                    logger.info(f"Input {i} del modelo: {inp.shape}")
            
            # Realizar la predicción
            y = model.predict(input_tensor, verbose=0).squeeze()
            logger.info(f"Resultado de la predicción: {y}, tipo: {type(y)}, dims: {np.ndim(y)}")
            
            # Binario: escalar | Multiclase: vector softmax
            if np.ndim(y) == 0:
                probs = np.array([1.0 - float(y), float(y)], dtype=np.float32)
            else:
                probs = np.array(y, dtype=np.float32)
            
            logger.info(f"Probabilidades finales: {probs}")
            self._last_probs = probs
            self._last_label = int(np.argmax(probs))
            logger.info(f"Label predicho: {self._last_label}")
            return probs
        except Exception as e:
            logger.error(f"Error en inferencia del modelo Keras: {e}", exc_info=True)
            # Devolver probabilidades por defecto: [1.0, 0.0] para que is_valid_ec_id devuelva False
            probs = np.array([1.0, 0.0], dtype=np.float32)
            self._last_probs = probs
            self._last_label = 0
            return probs

    def is_valid_ec_id(self, image_bgr: np.ndarray) -> tuple[bool, float]:
        probs = self._infer(image_bgr)
        top_score = float(np.max(probs))
        # Este booleano usa SOLO el umbral de score; tu servicio impone la política (labels 0,1,2)
        return (top_score >= self.th, top_score)

    # === Método que tu servicio busca (para _try_get_predicted_label) ===
    def predict_label(self, image_bgr: Optional[np.ndarray] = None) -> int:
        if image_bgr is not None or self._last_label is None:
            self._infer(image_bgr)
        return int(self._last_label)

    # Útil para diagnostics (top-k en el servicio)
    @property
    def last_probs(self) -> Optional[np.ndarray]:
        return self._last_probs
