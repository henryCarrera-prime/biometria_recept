# biometria/infrastructure/similarity/rekognition_adapter.py
import boto3, io, cv2
from PIL import Image

class RekognitionMatcher:
    def __init__(self, region: str, similarity_th: float):
        self.client = boto3.client("rekognition", region_name=region)
        self.similarity_th = similarity_th

    def _to_jpeg_bytes(self, img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    def compare(self, probe_bgr, reference_bgr) -> float:
        probe = self._to_jpeg_bytes(probe_bgr)
        ref = self._to_jpeg_bytes(reference_bgr)
        resp = self.client.compare_faces(
            SourceImage={"Bytes": probe},
            TargetImage={"Bytes": ref},
            SimilarityThreshold=self.similarity_th
        )
        return max([m["Similarity"] for m in resp.get("FaceMatches", [])], default=0.0)
