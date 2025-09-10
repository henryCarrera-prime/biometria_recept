# biometria/presentation/api.py
import os, json, uuid, datetime
from typing import List, Dict, Any, Optional
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import FileResponse
from biometria.infrastructure.config import build_verify_service_auto, get_thresholds
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from .schemas import (
    BiometricsVerifyRequestSerializer,
    BiometricsVerifyResponseSerializer,
    BiometricsFlowSerializer,
    TraceResponseSerializer
)
FLOW_LOG_DIR = os.getenv("FLOW_LOG_DIR", os.path.join(os.getcwd(), "biometria_flows"))

def _ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def _now_iso():
    return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat()

def _safe_write_json(path: str, data: Dict[str, Any]):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _read_json_file(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _append_general_index(uuid_proceso: str, item: Dict[str, Any]):
    """
    Mantiene un índice por proceso general:
      biometria_flows/trace_<uuidProceso>.json

    Estructura:
    {
      "uuid_proceso": "<uuidProceso>",
      "count": <N>,
      "items": [ { resumen por ejecución }, ... ]  // más nuevo primero
    }
    """
    _ensure_dir(FLOW_LOG_DIR)
    idx_path = os.path.join(FLOW_LOG_DIR, f"trace_{uuid_proceso}.json")
    idx = _read_json_file(idx_path) or {"uuid_proceso": uuid_proceso, "count": 0, "items": []}

    # Insertar al inicio (orden: más reciente primero)
    idx["items"].insert(0, item)
    idx["count"] = len(idx["items"])
    _safe_write_json(idx_path, idx)

def _scan_flows_by_uuid_proceso(uuid_proceso: str) -> List[Dict[str, Any]]:
    """
    Fallback por si no existiera el índice: recorre todos los .txt.
    (Úsalo solo si no hay trace_<uuid>.json; puede ser más lento si hay muchos archivos).
    Devuelve una lista de resúmenes (más nuevos primero).
    """
    _ensure_dir(FLOW_LOG_DIR)
    out: List[Dict[str, Any]] = []
    for name in os.listdir(FLOW_LOG_DIR):
        if not name.endswith(".txt"):
            continue
        data = _read_json_file(os.path.join(FLOW_LOG_DIR, name))
        if not data:
            continue
        if data.get("uuid_proceso") != uuid_proceso:
            continue

        # Construir resumen coherente con el índice
        item = {
            "uuid_proceso_biometria": data.get("uuid_proceso_biometria"),
            "status": data.get("result", {}).get("status") or data.get("status") or "unknown",
            "message": (data.get("result", {}) or {}).get("message") or data.get("message"),
            "evaluation_pct": (data.get("result", {}) or {}).get("evaluation_pct"),
            "similarity": (data.get("result", {}) or {}).get("similarity"),
            "is_live": (data.get("result", {}) or {}).get("is_live"),
            "is_match": (data.get("result", {}) or {}).get("is_match"),
            "liveness_score": (data.get("result", {}) or {}).get("liveness_score"),
            "reference_uri": data.get("reference_uri"),
            "frames_uri": data.get("frames_uri"),
            "frames_count": data.get("frames_count"),
            "started_at_utc": data.get("started_at_utc"),
            "finished_at_utc": data.get("finished_at_utc"),
        }
        out.append(item)

    # Ordenar por finished_at_utc descendente
    out.sort(key=lambda x: x.get("finished_at_utc") or "", reverse=True)
    return out

def _paginate(items: List[Dict[str, Any]], offset: int, limit: int) -> Dict[str, Any]:
    total = len(items)
    sliced = items[offset: offset + limit]
    return {
        "items": sliced,
        "page": {
            "offset": offset,
            "limit": limit,
            "returned": len(sliced),
            "total": total,
            "has_more": (offset + limit) < total
        }
    }

class VerifyBiometricsAPIView(APIView):
    """
    POST /api/biometrics/verify

    Body (S3 o local):
    {
      "uuidProceso": "04205d9c-1439-4a6e-a06c-21012a4ea744",
      "referenceImageUrl": "s3://.../carpeta-referencia/",
      "framesIndexUrl":    "s3://.../carpeta-frames/"
    }
    """
    @swagger_auto_schema(
        operation_summary="Verificar biometría (S3 o local)",
        operation_description=(
            "Ejecuta el flujo de verificación.\n\n"
            "- `referenceImageUrl` y `framesIndexUrl` aceptan rutas locales o URIs/URLs S3.\n"
            "- Retorna un nuevo `uuid_proceso_biometria` y la evaluación.\n"
            "- También escribe un `.txt` con JSON del flujo para auditoría."
        ),
        request_body=BiometricsVerifyRequestSerializer,
        responses={200: BiometricsVerifyResponseSerializer},
        tags=["Biometría"]
    )
    def post(self, request):
        try:
            body = request.data or {}
            uuid_proceso = (body.get("uuidProceso") or "").strip()
            reference_dir = (body.get("referenceImageUrl") or "").strip()
            frames_dir = (body.get("framesIndexUrl") or "").strip()

            if not uuid_proceso:
                return Response({"status": "false", "message": "Falta uuidProceso", "data": []}, status=200)
            if not reference_dir:
                return Response({"status": "false", "message": "Falta referenceImageUrl", "data": []}, status=200)
            if not frames_dir:
                return Response({"status": "false", "message": "Falta framesIndexUrl", "data": []}, status=200)

            uuid_biometria = str(uuid.uuid4())
            svc = build_verify_service_auto()
            thresholds = get_thresholds()

            # Contar frames de forma segura
            try:
                frames_list = svc.frames_repo.list_frames(frames_dir)
                frames_count = len(frames_list)
            except Exception:
                frames_list = []
                frames_count = 0

            started_at = _now_iso()
            result = svc.execute(uuid_proceso, reference_dir, frames_dir)

            passed = bool(result.is_live and result.is_match and (result.evaluation_pct >= thresholds.similarity))
            status_str = "success" if passed else "false"
            msg = (
                f"Evaluación {result.evaluation_pct:.2f}% (≥ {thresholds.similarity:.2f}%) - aprobado."
                if passed else
                f"Evaluación {result.evaluation_pct:.2f}% (< {thresholds.similarity:.2f}%) - no aprobado."
            )

            # --------- Guardar flujo en TXT con JSON ----------
            _ensure_dir(FLOW_LOG_DIR)
            flow = {
                "uuid_proceso_biometria": uuid_biometria,
                "uuid_proceso": uuid_proceso,
                "started_at_utc": started_at,
                "finished_at_utc": _now_iso(),
                "reference_uri": reference_dir,
                "frames_uri": frames_dir,
                "frames_count": frames_count,
                "thresholds": {
                    "similarity": thresholds.similarity,
                    "live": thresholds.live,
                    "luxand": thresholds.luxand
                },
                "result": {
                    "status": status_str,
                    "message": msg,
                    "is_live": result.is_live,
                    "is_match": result.is_match,
                    "liveness_score": round(result.liveness_score, 4),
                    "similarity": round(result.similarity, 2),
                    "evaluation_pct": round(result.evaluation_pct, 2)
                }
            }
            log_path = os.path.join(FLOW_LOG_DIR, f"{uuid_biometria}.txt")
            _safe_write_json(log_path, flow)

            # --------- NUEVO: actualizar índice por uuidProceso ----------
            summary_item = {
                "uuid_proceso_biometria": uuid_biometria,
                "status": status_str,
                "message": msg,
                "evaluation_pct": round(result.evaluation_pct, 2),
                "similarity": round(result.similarity, 2),
                "is_live": result.is_live,
                "is_match": result.is_match,
                "liveness_score": round(result.liveness_score, 4),
                "reference_uri": reference_dir,
                "frames_uri": frames_dir,
                "frames_count": frames_count,
                "started_at_utc": started_at,
                "finished_at_utc": flow["finished_at_utc"]
            }
            _append_general_index(uuid_proceso, summary_item)

            return Response({
                "status": status_str,
                "message": msg,
                "data": [{
                    "uuid_proceso_biometria": uuid_biometria,
                    "evaluacion": round(result.evaluation_pct, 2)
                }]
            }, status=200)

        except Exception as ex:
            uuid_biometria = locals().get("uuid_biometria") or str(uuid.uuid4())
            _ensure_dir(FLOW_LOG_DIR)
            error_log = {
                "uuid_proceso_biometria": uuid_biometria,
                "error": str(ex),
                "finished_at_utc": _now_iso()
            }
            try:
                _safe_write_json(os.path.join(FLOW_LOG_DIR, f"{uuid_biometria}.txt"), error_log)
            except Exception:
                pass

            return Response({
                "status": "false",
                "message": "Error no controlado en biometría",
                "data": [{"uuid_proceso_biometria": uuid_biometria}]
            }, status=200)

# ---------- Consultar por uuid_proceso_biometria ----------
download_param = openapi.Parameter(
    "download",
    openapi.IN_QUERY,
    description="Si es true/1, descarga el .txt original como attachment.",
    type=openapi.TYPE_BOOLEAN
)
class ConsultBiometricsAPIView(APIView):
    """
    GET /api/biometrics/verify/<uuid_proceso_biometria>[?download=1]
    - Sin query: retorna el JSON guardado en el .txt (o raw si no es JSON).
    - Con ?download=1: descarga el .txt original (attachment).
    """
    @swagger_auto_schema(
        operation_summary="Consultar ejecución por uuid_proceso_biometria",
        operation_description=(
            "Devuelve el JSON del flujo guardado (y enlaces `_links`).\n"
            "Si `download=1`, descarga el `.txt` original (text/plain)."
        ),
        manual_parameters=[download_param],
        responses={
            200: BiometricsFlowSerializer,
            # Variante de descarga (OpenAPI 2.0): type=file
            "200 (download)": openapi.Schema(type=openapi.TYPE_FILE, description="Archivo .txt del flujo"),
            404: "No existe el proceso solicitado."
        },
        tags=["Biometría"]
    )
    def get(self, request, uuid_biometria: str):
        _ensure_dir(FLOW_LOG_DIR)
        path = os.path.join(FLOW_LOG_DIR, f"{uuid_biometria}.txt")
        if not os.path.exists(path):
            return Response({"detail": "No existe el proceso solicitado."}, status=status.HTTP_404_NOT_FOUND)

        # Modo descarga
        download = (request.query_params.get("download") or "false").lower() in ("1", "true", "yes")
        if download:
            # Stream del archivo como attachment
            return FileResponse(
                open(path, "rb"),
                as_attachment=True,
                filename=f"{uuid_biometria}.txt",
                content_type="text/plain; charset=utf-8"
            )

        # Modo JSON + enlaces
        data = _read_json_file(path)
        if data is None:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            payload = {"uuid_proceso_biometria": uuid_biometria, "raw": text}
        else:
            payload = data

        # Enlaces útiles (self + descarga)
        payload.setdefault("_links", {})
        payload["_links"]["self"] = request.build_absolute_uri()
        # Aseguramos que la URL tenga el query download=1
        base_url = request.build_absolute_uri().split("?", 1)[0]
        payload["_links"]["download"] = f"{base_url}?download=1"

        return Response(payload, status=200)


# ---------- Trazabilidad por uuidProceso ----------
expand_param = openapi.Parameter(
    "expand", openapi.IN_QUERY, description="Si true, retorna el JSON completo de cada ejecución.", type=openapi.TYPE_BOOLEAN
)
offset_param = openapi.Parameter(
    "offset", openapi.IN_QUERY, description="Desplazamiento de paginación.", type=openapi.TYPE_INTEGER, default=0
)
limit_param = openapi.Parameter(
    "limit", openapi.IN_QUERY, description="Tamaño de página.", type=openapi.TYPE_INTEGER, default=50
)
class TraceGeneralProcessAPIView(APIView):
    """
    GET /api/biometrics/trace/<uuid_proceso>?expand=false&offset=0&limit=50

    - Retorna TODAS las ejecuciones del proceso (cada una con su uuid_proceso_biometria).
    - Por defecto devuelve un resumen por ítem (rápido). Con expand=true, abre cada .txt para detalle completo.
    - Orden: más nuevo primero.
    """
    @swagger_auto_schema(
        operation_summary="Trazabilidad por uuidProceso",
        operation_description=(
            "Lista todas las ejecuciones realizadas para un proceso general (`uuidProceso`).\n"
            "Orden: más nuevo primero. Usa `expand=true` para ver cada flujo completo."
        ),
        manual_parameters=[expand_param, offset_param, limit_param],
        responses={200: TraceResponseSerializer},
        tags=["Biometría"]
    )
    def get(self, request, uuid_proceso: str):
        expand = (request.query_params.get("expand") or "false").lower() in ("1", "true", "yes")
        try:
            offset = int(request.query_params.get("offset") or 0)
            limit = int(request.query_params.get("limit") or 50)
        except ValueError:
            return Response({"detail": "offset/limit inválidos"}, status=status.HTTP_400_BAD_REQUEST)

        _ensure_dir(FLOW_LOG_DIR)
        idx_path = os.path.join(FLOW_LOG_DIR, f"trace_{uuid_proceso}.json")
        if os.path.exists(idx_path):
            idx = _read_json_file(idx_path) or {"items": []}
            items = idx.get("items", [])
        else:
            # Fallback lento, escanear .txt
            items = _scan_flows_by_uuid_proceso(uuid_proceso)

        # Si expand=true, reemplazamos cada resumen por el JSON completo del flujo
        if expand:
            expanded = []
            for it in items:
                uuid_bio = it.get("uuid_proceso_biometria")
                if not uuid_bio:
                    continue
                flow_path = os.path.join(FLOW_LOG_DIR, f"{uuid_bio}.txt")
                full = _read_json_file(flow_path)
                if full:
                    expanded.append(full)
                else:
                    expanded.append({"uuid_proceso_biometria": uuid_bio, "detail": "no disponible"})
            items = expanded

        sliced = _paginate(items, offset, limit)
        return Response({
            "uuid_proceso": uuid_proceso,
            "count": len(items) if not expand else len(items),
            **sliced
        }, status=200)
