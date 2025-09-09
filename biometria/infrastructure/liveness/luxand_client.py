import os
import cv2
import requests
import logging
import time

logger = logging.getLogger("biometria.verify")

class LuxandClient:
    """
    Cliente para Luxand Cloud Liveness (foto).
    Devuelve un score 0..1 (probabilidad de 'alive').
    - Usa v2 por defecto.
    - Si v2 responde 404, hace fallback a /photo/liveness (estable).
    - Si no hay token, queda inactivo (is_active=False) y score=0.0.
    """
    def __init__(self, token: str | None = None, api_url: str | None = None, timeout: int = 10, retries: int = 1):
        # Token desde parámetro o .env
        self.token = (token or os.getenv("LUXAND_TOKEN", "")).strip()
        self.is_active = bool(self.token)

        # URL desde parámetro o .env, con fallback por defecto (v2 → estable)
        env_url = (api_url or os.getenv("LUXAND_API_URL", "")).strip()
        self.primary_url = env_url if env_url else "https://api.luxand.cloud/photo/liveness/v2"
        self.fallback_url = "https://api.luxand.cloud/photo/liveness"

        self.timeout = timeout
        self.retries = max(0, int(retries))  # reintentos suaves para 429/5xx

        if not self.is_active:
            logger.info({"event": "luxand_inactive_no_token"})
        else:
            logger.info({"event": "luxand_init", "primary_url": self.primary_url})

    def _post_photo(self, url: str, jpg_bytes: bytes):
        headers = {"token": self.token}
        files = {"photo": ("frame.jpg", jpg_bytes, "image/jpeg")}
        return requests.post(url, headers=headers, files=files, timeout=self.timeout)

    def _parse_score(self, data: dict) -> float:
        # Acepta distintos formatos de respuesta.
        if "probability" in data:
            return float(data.get("probability", 0.0))
        if "liveness" in data:
            return float(data.get("liveness", 0.0))
        if "alive" in data:
            return 1.0 if data.get("alive") else 0.0
        return 0.0

    def score(self, face_bgr) -> float:
        if not self.is_active:
            return 0.0

        ok, buf = cv2.imencode(".jpg", face_bgr)
        if not ok:
            logger.info({"event": "luxand_encode_fail"})
            return 0.0
        jpg_bytes = buf.tobytes()

        urls_to_try = [self.primary_url]

        # Si la primary es v2, planifica fallback a estable ante 404
        if self.primary_url.endswith("/v2"):
            urls_to_try.append(self.fallback_url)

        for idx, url in enumerate(urls_to_try):
            attempt = 0
            while True:
                try:
                    resp = self._post_photo(url, jpg_bytes)
                    # Fallback manual si la primary devuelve 404
                    if resp.status_code == 404 and idx == 0 and url.endswith("/v2"):
                        logger.info({"event": "luxand_fallback_404", "from": url, "to": urls_to_try[idx+1]})
                        break  # sale del while y pasa al siguiente url

                    # Reintento básico para 429/5xx
                    if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.retries:
                        attempt += 1
                        wait = min(1.5 * attempt, 3.0)
                        logger.info({"event": "luxand_retry", "status": resp.status_code, "attempt": attempt, "wait": wait})
                        time.sleep(wait)
                        continue

                    resp.raise_for_status()
                    data = resp.json()
                    score = self._parse_score(data)
                    score = max(0.0, min(1.0, float(score)))
                    logger.info({"event": "luxand_response", "api_url": url, "score": round(score, 4), "keys": list(data.keys())})
                    return score

                except requests.RequestException as e:
                    # Si es el último intento de este URL y/o último URL, loggea y sigue/finaliza
                    logger.info({"event": "luxand_http_error", "api_url": url, "error": str(e), "attempt": attempt})
                    if attempt < self.retries:
                        attempt += 1
                        wait = min(1.5 * attempt, 3.0)
                        time.sleep(wait)
                        continue
                    break  # pasa al siguiente URL si existe

                except Exception as e:
                    logger.info({"event": "luxand_error", "api_url": url, "error": str(e)})
                    break  # pasa al siguiente URL si existe

        # Si todas las rutas fallaron, devolvemos 0.0 (no rompemos el endpoint)
        return 0.0
