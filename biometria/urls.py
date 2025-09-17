# biometria/urls.py
from django.urls import path
from biometria.presentation.api import (
    VerifyBiometricsAPIView,
    ConsultBiometricsAPIView,
    TraceGeneralProcessAPIView,
    VerifyCedulaAPIView,
    ConsultCedulaAPIView,
    TraceCedulaProcessAPIView
)

app_name = "biometria"

urlpatterns = [
    path('biometrics/verify', VerifyBiometricsAPIView.as_view(), name='verify'),
    path('biometrics/verify/<str:uuid_biometria>', ConsultBiometricsAPIView.as_view(), name='verify-detail'),
    path('biometrics/trace/<str:uuid_proceso>', TraceGeneralProcessAPIView.as_view(), name='trace'),
    #Cedula
    path('biometrics/cedula/verify', VerifyCedulaAPIView.as_view(), name='cedula-verify'),
    path('biometrics/cedula/verify/<str:uuid_proceso_cedula>', ConsultCedulaAPIView.as_view(), name='cedula-verify-detail'),
    path('biometrics/cedula/trace/<str:uuidProceso>', TraceCedulaProcessAPIView.as_view(), name='cedula-trace'),
]
