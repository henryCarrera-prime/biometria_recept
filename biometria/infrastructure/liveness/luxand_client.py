# biometria/infrastructure/liveness/luxand_client.py
import requests, io, cv2
from PIL import Image

class LuxandClient:
    def __init__(self, token: str):
        self.token = token

    def score(self, img_bgr) -> float:
        if not self.token: return 0.0
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=90)
        files = {"photo": ("frame.jpg", buf.getvalue(), "image/jpeg")}
        try:
            r = requests.post(
                "https://api.luxand.cloud/liveness",
                headers={"Authorization": f"Bearer {self.token}"},
                files=files, timeout=12
            )
            if r.status_code != 200: return 0.0
            print("Luxand response:", r.json())
            return float(r.json().get("liveness", 0.0))
        except Exception:
            return 0.0
