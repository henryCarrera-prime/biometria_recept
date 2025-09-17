# biometria/infrastructure/id_classifier/keras_ecuador_id_classifier.py
import os, threading, numpy as np, cv2
import tensorflow as tf
import re

_model = None
_model_lock = threading.Lock()

def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(raw))
    return float(m.group(0)) if m else float(default)

def _parse_size():
    s = os.getenv("ECU_ID_INPUT_SIZE", "224x224").lower().replace(" ", "")
    try:
        w, h = s.split("x")
        return int(w), int(h)
    except Exception:
        return (224, 224)

def _load_tf_model(path: str):
    # 1) Intenta carga "limpia" con tf.keras
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        msg = str(e)
        # 2) Solo si falla por 'groups' en DepthwiseConv2D, aplica parche puntual
        if "DepthwiseConv2D" in msg and "Unrecognized keyword" in msg and "groups" in msg:
            class DepthwiseConv2DPatched(tf.keras.layers.DepthwiseConv2D):
                @classmethod
                def from_config(cls, cfg):
                    cfg.pop("groups", None)
                    return super().from_config(cfg)
            return tf.keras.models.load_model(
                path, compile=False, custom_objects={"DepthwiseConv2D": DepthwiseConv2DPatched}
            )
        # Repropaga cualquier otro error (para verlo claro)
        raise

def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                path = os.getenv("ECU_ID_MODEL_PATH", "/app/models/keras_model.h5")
                _model = _load_tf_model(path)
    return _model

class KerasEcuadorIdClassifier:
    def __init__(self):
        self.size = _parse_size()
        self.th = _env_float("ECU_ID_THRESHOLD", 0.85)

    def _prep(self, img_bgr: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
        img = img.astype("float32") / 255.0
        return np.expand_dims(img, axis=0)  # (1,H,W,3)

    def is_valid_ec_id(self, image_bgr: np.ndarray) -> tuple[bool, float]:
        model = _get_model()
        y = model.predict(self._prep(image_bgr), verbose=0).squeeze()
        # Binario: escalar | Multiclase: usa máx (ajusta si conoces la(s) clase(s) positivas)
        score = float(y) if np.ndim(y) == 0 else float(np.max(y))
        return (score >= self.th, score)
