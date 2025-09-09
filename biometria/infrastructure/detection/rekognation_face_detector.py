import os
import io
import cv2
import boto3
import logging
from typing import Optional, Dict, List, Tuple
from PIL import Image

logger = logging.getLogger("biometria.verify")

def _to_jpg_bytes(img_bgr) -> bytes:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=90)
    return buf.getvalue()

def _relbox_to_pixels(rel_box: Dict[str, float], width: int, height: int) -> Tuple[int,int,int,int]:
    x1 = int(round(rel_box["Left"] * width))
    y1 = int(round(rel_box["Top"] * height))
    x2 = int(round((rel_box["Left"] + rel_box["Width"]) * width))
    y2 = int(round((rel_box["Top"] + rel_box["Height"]) * height))
    x1 = max(0, min(x1, width-1)); y1 = max(0, min(y1, height-1))
    x2 = max(0, min(x2, width-1)); y2 = max(0, min(y2, height-1))
    if x2 <= x1: x2 = min(width-1, x1+1)
    if y2 <= y1: y2 = min(height-1, y1+1)
    return (x1, y1, x2, y2)

def _landmarks_to_pixels(landmarks: List[Dict[str, float]], width: int, height: int) -> List[Tuple[int,int]]:
    pts = []
    for lm in landmarks or []:
        x = int(round(lm["X"] * width)); y = int(round(lm["Y"] * height))
        x = max(0, min(x, width-1)); y = max(0, min(y, height-1))
        pts.append((x, y))
    return pts

class RekognitionFaceDetector:
    """
    FaceDetector con AWS Rekognition DetectFaces.
    Retorna:
      {
        "bbox": (x1,y1,x2,y2),
        "landmarks": [(x,y), ...] | None,
        "confidence": float(0..100),
        "area_rel": float(0..1),
        "pose": {"Yaw": float, "Pitch": float, "Roll": float} | {},
        "quality": {"Sharpness": float(0..100), "Brightness": float(0..100)} | {}
      }
    o None si no hay rostro válido.
    """
    def __init__(
        self,
        region: Optional[str] = None,
        min_confidence: float = 80.0,
        min_face_rel_size: float = 0.05,  # 5% del frame
        attributes: List[str] = None,     # ["ALL"] para landmarks/pose/quality
    ):
        self.client = boto3.client("rekognition", region_name=region or os.getenv("AWS_REGION", "us-east-1"))
        self.min_confidence = float(min_confidence)
        self.min_face_rel_size = float(min_face_rel_size)
        self.attributes = attributes or ["ALL"]

    def detect(self, img_bgr) -> Optional[dict]:
        try:
            h, w = img_bgr.shape[:2]
            jpg = _to_jpg_bytes(img_bgr)
            resp = self.client.detect_faces(Image={"Bytes": jpg}, Attributes=self.attributes)
            faces = resp.get("FaceDetails", []) or []
            if not faces:
                return None

            # filtra por confianza y tamaño relativo
            candidates = []
            for f in faces:
                conf = float(f.get("Confidence", 0.0))
                bbox_rel = f.get("BoundingBox")
                if conf < self.min_confidence or not bbox_rel:
                    continue
                area_rel = (bbox_rel["Width"] * bbox_rel["Height"])
                if area_rel < self.min_face_rel_size:
                    continue
                candidates.append(f)

            if not candidates:
                return None

            # elige la cara más grande
            def face_area(f): 
                b = f["BoundingBox"]; return b["Width"] * b["Height"]
            best = max(candidates, key=face_area)

            bbox = _relbox_to_pixels(best["BoundingBox"], w, h)
            landmarks_px = _landmarks_to_pixels(best.get("Landmarks"), w, h)
            confidence = float(best.get("Confidence", 0.0))
            area_rel = float(best["BoundingBox"]["Width"] * best["BoundingBox"]["Height"])
            pose = best.get("Pose") or {}
            quality = best.get("Quality") or {}

            logger.info({
                "event": "rek_face_detect",
                "faces_total": len(faces),
                "candidates": len(candidates),
                "chosen_area_pct": round(100.0 * area_rel, 2),
                "confidence": round(confidence, 2),
                "pose": {k: round(float(pose.get(k, 0.0)), 1) for k in ("Yaw","Pitch","Roll")}
            })

            return {
                "bbox": bbox,
                "landmarks": landmarks_px if landmarks_px else None,
                "confidence": confidence,
                "area_rel": area_rel,
                "pose": pose,
                "quality": quality
            }
        except Exception as e:
            logger.info({"event": "rek_face_detect_error", "error": str(e)})
            return None
