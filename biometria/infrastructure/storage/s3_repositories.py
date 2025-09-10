# biometria/infrastructure/storage/s3_repositories.py
from __future__ import annotations
from typing import List, Optional, Tuple
import os, io, re
import boto3
import numpy as np
import cv2

from .local_repositories import IMG_EXT, load_local_image, list_images_in_dir
from .local_repositories import LocalReferenceRepository, LocalFramesRepository

# ---------------------------
# Utilidades para URIs de S3
# ---------------------------
def is_s3_uri(uri: str) -> bool:
    u = (uri or "").strip().lower()
    return u.startswith("s3://") or ".amazonaws.com/" in u

def parse_s3_uri(uri: str) -> Tuple[str, str]:
    """
    Acepta formatos:
      - s3://bucket/prefix/opcional/
      - https://<bucket>.s3.<region>.amazonaws.com/prefix/...
      - https://s3.<region>.amazonaws.com/<bucket>/prefix/...
    Retorna (bucket, prefix) sin '/' inicial.
    """
    u = (uri or "").strip()

    if u.startswith("s3://"):
        rest = u[5:]  # quita 's3://'
        bucket, _, prefix = rest.partition("/")
        return bucket, prefix.lstrip("/")

    m = re.match(r"https?://([^./]+)\.s3[.-][^/]+\.amazonaws\.com/(.+)", u)
    if m:
        return m.group(1), m.group(2).lstrip("/")

    m = re.match(r"https?://s3[.-][^/]+\.amazonaws\.com/([^/]+)/(.+)", u)
    if m:
        return m.group(1), m.group(2).lstrip("/")

    raise ValueError(f"URI S3 no reconocida: {uri}")

_s3_client = None
def s3_client():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3", region_name=os.getenv("AWS_REGION"))
    return _s3_client

def _list_images_s3(bucket: str, prefix: str) -> List[str]:
    cli = s3_client()
    token = None
    out: List[str] = []
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = cli.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(IMG_EXT):
                out.append(f"s3://{bucket}/{key}")
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return sorted(out)

def _get_image_s3(bucket: str, key: str) -> Optional[np.ndarray]:
    cli = s3_client()
    try:
        obj = cli.get_object(Bucket=bucket, Key=key)
        data = obj["Body"].read()
        arr = np.frombuffer(data, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None

# ---------------------------
# Repositorios S3 “puros”
# ---------------------------
class S3ReferenceRepository:
    def fetch_reference(self, uri: str) -> Optional[np.ndarray]:
        bucket, prefix = parse_s3_uri(uri)
        imgs = _list_images_s3(bucket, prefix)
        if not imgs:
            return None
        # Preferir archivo cuyo nombre empiece por "reference."
        pick = next((p for p in imgs if p.split("/")[-1].lower().startswith("reference.")), imgs[0])
        key = pick.split("/", 3)[3]  # s3://bucket/<key>
        return _get_image_s3(bucket, key)

class S3FramesRepository:
    def list_frames(self, uri: str) -> List[str]:
        bucket, prefix = parse_s3_uri(uri)
        return _list_images_s3(bucket, prefix)

    def fetch_frame(self, uri: str) -> Optional[np.ndarray]:
        if uri.lower().startswith("s3://"):
            # s3://bucket/key...
            _, _, rest = uri.partition("s3://")
            bucket, _, key = rest.partition("/")
            return _get_image_s3(bucket, key)
        # URLs https de S3
        bucket, prefix = parse_s3_uri(uri)
        return _get_image_s3(bucket, prefix)

# -------------------------------------------------
# Repos “Smart” que aceptan LOCAL y S3 transparentes
# -------------------------------------------------
class SmartReferenceRepository:
    def __init__(self):
        self._local = LocalReferenceRepository()
        self._s3 = S3ReferenceRepository()

    def fetch_reference(self, uri: str) -> Optional[np.ndarray]:
        return self._s3.fetch_reference(uri) if is_s3_uri(uri) else self._local.fetch_reference(uri)

class SmartFramesRepository:
    def __init__(self):
        self._local = LocalFramesRepository()
        self._s3 = S3FramesRepository()

    def list_frames(self, uri: str) -> List[str]:
        return self._s3.list_frames(uri) if is_s3_uri(uri) else self._local.list_frames(uri)

    def fetch_frame(self, uri: str) -> Optional[np.ndarray]:
        return self._s3.fetch_frame(uri) if is_s3_uri(uri) else self._local.fetch_frame(uri)
