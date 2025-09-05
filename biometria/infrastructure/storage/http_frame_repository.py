# biometria/infrastructure/storage/http_frame_repository.py
from typing import List, Optional
import requests, numpy as np, cv2

class HttpFrameRepository:
    def list_frame_urls(self, index_url: str | None, urls: List[str] | None) -> List[str]:
        if urls: return urls
        if not index_url: return []
        r = requests.get(index_url, timeout=10)
        r.raise_for_status()
        data = r.json()
        return [u for u in data if isinstance(u, str)]

    def fetch_frame(self, url: str) -> Optional[np.ndarray]:
        try:
            r = requests.get(url, timeout=10)
            if r.status_code != 200: return None
            arr = np.frombuffer(r.content, np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return None
