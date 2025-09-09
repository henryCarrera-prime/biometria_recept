import os
import cv2
import requests
import logging
import time

logger = logging.getLogger("biometria.verify")

class LuxandClient:
    """
    Cliente para Luxand Cloud Liveness (foto).
    - Usa /photo/liveness/v2 por defecto (devuelve {"status","score","result"}).
    - Fallback a /photo/liveness (formatos antiguos: "probability"/"liveness"/"alive").
    - Guarda last_response_json para auditoría.
    - score(img_bgr) -> float 0..1
    - evaluate(img_bgr) -> {"ok": bool, "score": float, "label": str|None}
    """
    def __init__(self, token: str | None = None, api_url: str | None = None, timeout: int = 10, retries: int = 1):
        self.token = (token or os.getenv("LUXAND_TOKEN", "")).strip()
        self.is_active = bool(self.token)

        env_url = (api_url or os.getenv("LUXAND_API_URL", "")).strip()
        self.primary_url = env_url if env_url else "https://api.luxand.cloud/photo/liveness/v2"
        self.fallback_url = "https://api.luxand.cloud/photo/liveness"

        self.timeout = timeout
        self.retries = max(0, int(retries))

        # para inspección desde el servicio
        self.last_status = None
        self.last_reason = None
        self.last_payload_info = None
        self.last_response_json = None

        if not self.is_active:
            logger.info({"event": "luxand_inactive_no_token"})
        else:
            logger.info({"event": "luxand_init", "primary_url": self.primary_url})

    def _post_photo(self, url: str, jpg_bytes: bytes):
        headers = {"token": self.token}
        files = {"photo": ("frame.jpg", jpg_bytes, "image/jpeg")}
        return requests.post(url, headers=headers, files=files, timeout=self.timeout)

    @staticmethod
    def _extract_score_and_label(data: dict):
        """
        Acepta múltiples formatos:
          v2: {"status":"success","score":0.98,"result":"real"}
          legacy: {"probability":..} o {"liveness":..} o {"alive":bool}
        """
        if not isinstance(data, dict):
            return 0.0, None

        # v2
        if "score" in data:
            try:
                return float(data.get("score", 0.0) or 0.0), (data.get("result") or data.get("label"))
            except Exception:
                pass

        # legacy
        if "probability" in data:
            try:
                return float(data.get("probability", 0.0) or 0.0), (data.get("result") or data.get("label"))
            except Exception:
                pass
        if "liveness" in data:
            try:
                return float(data.get("liveness", 0.0) or 0.0), (data.get("result") or data.get("label"))
            except Exception:
                pass
        if "alive" in data:
            try:
                return (1.0 if data.get("alive") else 0.0), (data.get("result") or data.get("label"))
            except Exception:
                pass

        return 0.0, (data.get("result") or data.get("label"))

    def _encode_jpg(self, img_bgr, quality=92) -> bytes:
        # JPEG baseline, optimize ON; evita progresivo por compat (opcional)
        encode_params = [
            int(cv2.IMWRITE_JPEG_QUALITY), int(quality),
            int(cv2.IMWRITE_JPEG_OPTIMIZE), 1,
        ]
        ok, buf = cv2.imencode(".jpg", img_bgr, encode_params)
        if not ok:
            logger.info({"event": "luxand_jpeg_encode_failed", "quality": quality})
            return b""
        jpg_bytes = buf.tobytes()
        logger.info({"event": "luxand_jpeg_stats", "quality": quality, "bytes": len(jpg_bytes)})
        return jpg_bytes

    @staticmethod
    def _ensure_min_size(img_bgr, min_short=400):
        h, w = img_bgr.shape[:2]
        short = min(h, w)
        if short >= min_short:
            return img_bgr
        scale = float(min_short) / max(1, short)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    def _dump_attempt(self, img_bgr, tag, debug=True):
        if not debug:
            return
        try:
            out_dir = os.getenv("DEBUG_LUXAND_DIR", "./debug_luxand_sent")
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"sent_{tag}.jpg")
            cv2.imwrite(path, img_bgr)
            logger.info({"event": "luxand_dump_saved", "file": path})
        except Exception as e:
            logger.info({"event": "luxand_dump_error", "error": str(e)})

    def _post_and_parse(self, url, img_bgr, tag, debug=True):
        jpg = self._encode_jpg(img_bgr, quality=92)
        if not jpg:
            self.last_reason = "encode_error"
            return None, {}, None

        h, w = img_bgr.shape[:2]
        self.last_payload_info = (w, h, len(jpg), tag)
        logger.info({
            "event": "luxand_request",
            "api_url": url,
            "tag": tag,
            "img_size": f"{w}x{h}",
            "jpg_bytes": len(jpg),
            "aspect_ratio": round(w / h, 2) if h > 0 else 0,
            "min_dimension": min(w, h),
            "max_dimension": max(w, h)
        })
        self._dump_attempt(img_bgr, tag, debug=debug)

        try:
            resp = self._post_photo(url, jpg)

            # Fallback si /v2 no existe en el plan
            if resp.status_code == 404 and url.endswith("/v2"):
                logger.info({"event": "luxand_fallback_404", "from": url, "to": self.fallback_url})
                resp = self._post_photo(self.fallback_url, jpg)

            raw_text = ""
            try:
                raw_text = resp.text[:500]
            except Exception:
                pass
            logger.info({"event": "luxand_raw_response", "status": resp.status_code, "tag": tag, "raw": raw_text})

            self.last_status = resp.status_code
            resp.raise_for_status()
            data = {}
            try:
                data = resp.json()
            except Exception:
                pass
            # Guarda SIEMPRE el último JSON
            self.last_response_json = data

            score, label = self._extract_score_and_label(data)

            if score > 0.0:
                logger.info({"event": "luxand_response", "api_url": url, "tag": tag, "score": round(score, 4), "keys": list(data.keys())})
                return score, data, resp.status_code

            # Sin score numérico → log detallado de mensaje
            msg = (data.get("message") or "no_message")
            self.last_reason = msg
            logger.info({
                "event": "luxand_message_only",
                "api_url": url,
                "tag": tag,
                "message": msg,
                "keys": list(data.keys()),
                "status_code": resp.status_code,
                "response_data": data
            })
            return None, data, resp.status_code

        except requests.RequestException as e:
            self.last_reason = f"http_error: {str(e)}"
            logger.info({"event": "luxand_http_error", "api_url": url, "tag": tag, "error": str(e)})
            return None, {"exception": str(e)}, None
        except Exception as e:
            self.last_reason = f"client_error: {str(e)}"
            logger.info({"event": "luxand_error", "api_url": url, "tag": tag, "error": str(e)})
            return None, {"exception": str(e)}, None

    def score(self, img_bgr) -> float:
        """
        Devuelve un score 0..1. Intenta v2 y luego legacy.
        Reintenta ante 429/5xx si self.retries > 0.
        """
        if not self.is_active:
            return 0.0

        self.last_reason = None
        self.last_status = None
        self.last_payload_info = None
        self.last_response_json = None

        urls_to_try = [self.primary_url]
        if self.primary_url.endswith("/v2"):
            urls_to_try.append(self.fallback_url)

        # 1) intento con la imagen tal cual
        for url in urls_to_try:
            attempt = 0
            while True:
                s, data, status = self._post_and_parse(url, img_bgr, tag="crop", debug=True)
                if isinstance(s, float):
                    return max(0.0, min(1.0, s))
                if status in (429, 500, 502, 503, 504) and attempt < self.retries:
                    attempt += 1
                    wait = min(1.5 * attempt, 3.0)
                    logger.info({"event": "luxand_retry", "status": status, "attempt": attempt, "wait": wait})
                    time.sleep(wait)
                    continue
                break

        # 2) reintento con reescalado suave (lado corto >= 400)
        scaled = self._ensure_min_size(img_bgr, min_short=400)
        if scaled is not None and (scaled.shape[:2] != img_bgr.shape[:2]):
            for url in urls_to_try:
                attempt = 0
                while True:
                    s, data, status = self._post_and_parse(url, scaled, tag="crop_scaled", debug=True)
                    if isinstance(s, float):
                        return max(0.0, min(1.0, s))
                    if status in (429, 500, 502, 503, 504) and attempt < self.retries:
                        attempt += 1
                        wait = min(1.5 * attempt, 3.0)
                        logger.info({"event": "luxand_retry", "status": status, "attempt": attempt, "wait": wait})
                        time.sleep(wait)
                        continue
                    break

        if self.last_reason is None:
            self.last_reason = "no_score_no_message"
        return 0.0

    def evaluate(self, img_bgr):
        """
        Devuelve un dict homogéneo para el servicio:
        {"ok": bool, "score": float, "label": "real"/"spoof"/None}
        """
        s = self.score(img_bgr)
        lab = None
        if isinstance(self.last_response_json, dict):
            lab = self.last_response_json.get("result") or self.last_response_json.get("label")
        return {"ok": s > 0.0, "score": s, "label": lab}
