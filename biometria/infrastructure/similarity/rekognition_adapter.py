import os, io, cv2, boto3
from PIL import Image

class RekognitionMatcher:
    def __init__(self, region: str | None = None, similarity_th: float = 95.0):
        self.client = boto3.client("rekognition", region_name=region or os.getenv("AWS_REGION","us-east-1"))
        self.similarity_th = similarity_th

    def _to_jpg(self, img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    def compare(self, probe_bgr, reference_bgr) -> float:
        src = self._to_jpg(probe_bgr)
        tgt = self._to_jpg(reference_bgr)
        resp = self.client.compare_faces(
            SourceImage={"Bytes": src},
            TargetImage={"Bytes": tgt},
            SimilarityThreshold=self.similarity_th
        )
        return max([m["Similarity"] for m in resp.get("FaceMatches", [])], default=0.0)
