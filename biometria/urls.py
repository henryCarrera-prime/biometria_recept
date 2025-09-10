# biometria/urls.py
from django.urls import path
from biometria.presentation.api import (
    VerifyBiometricsAPIView,
    ConsultBiometricsAPIView,
    TraceGeneralProcessAPIView,
)

app_name = "biometria"

urlpatterns = [
    path('biometrics/verify', VerifyBiometricsAPIView.as_view(), name='verify'),
    path('biometrics/verify/<str:uuid_biometria>', ConsultBiometricsAPIView.as_view(), name='verify-detail'),
    path('biometrics/trace/<str:uuid_proceso>', TraceGeneralProcessAPIView.as_view(), name='trace'),
]
