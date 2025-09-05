# biometria/presentation/api.py
from rest_framework.views import APIView
from rest_framework.response import Response
from biometria.infrastructure.config import build_verify_service

class VerifyBiometricsAPIView(APIView):
    def post(self, request):
        body = request.data
        session_id = body.get("sessionId")
        ref_url = body.get("referenceImageUrl")
        idx_url = body.get("framesIndexUrl")
        urls = body.get("framesUrls")

        svc = build_verify_service()
        result = svc.execute(session_id, ref_url, idx_url, urls)

        # Respuesta en tu formato
        status_str = "success" if (result.is_live and result.is_match and result.evaluation_pct >= 95.0) else "false"
        return Response({
            "status": status_str,
            "message": result.message,
            "data": [{
                "uuid_proceso_biometria": session_id,
                "evaluacion": round(result.evaluation_pct, 2)
            }]
        })
