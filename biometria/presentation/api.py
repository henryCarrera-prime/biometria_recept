# biometria/presentation/api.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from biometria.infrastructure.config import build_verify_service_for_local_dirs

class VerifyBiometricsAPIView(APIView):
    """
    POST /api/biometrics/verify
    {
      "uuidProceso":"04205d9c-1439-4a6e-a06c-21012a4ea744",
      "referenceImageUrl": "C:stalin/carpeta/04205d9c-1439-4a6e-a06c-21012a4ea744",
      "framesIndexUrl":    "C:stalin/carpeta/04205d9c-1439-4a6e-a06c-21012a4ea744-frames"
    }
    """
    def post(self, request):
        uuid_proceso = request.data.get("uuidProceso")
        reference_dir = request.data.get("referenceImageUrl")
        frames_dir    = request.data.get("framesIndexUrl")

        if not uuid_proceso:
            return Response({"status":"false","message":"Falta uuidProceso","data":[]}, status=200)
        if not reference_dir:
            return Response({"status":"false","message":"Falta referenceImageUrl","data":[]}, status=200)
        if not frames_dir:
            return Response({"status":"false","message":"Falta framesIndexUrl","data":[]}, status=200)

        svc = build_verify_service_for_local_dirs()
        result = svc.execute(uuid_proceso, reference_dir, frames_dir)

        status_str = "success" if (result.is_live and result.is_match and result.evaluation_pct >= 95.0) else "false"
        return Response({
            "status": status_str,
            "message": result.message,
            "data": [{
                "uuid_proceso_biometria": uuid_proceso,
                "evaluacion": round(result.evaluation_pct, 2)
            }]
        }, status=200)
