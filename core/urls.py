# core/urls.py
from django.contrib import admin
from django.urls import path, include
from drf_yasg.views import get_schema_view
from drf_yasg import openapi
from rest_framework import permissions

schema_view = get_schema_view(
    openapi.Info(
        title="Biometría API",
        default_version="v1",
        description="Endpoints para verificación biométrica, consulta de ejecución y trazabilidad.",
        contact=openapi.Contact(email="soporte@tu-dominio.com"),
        license=openapi.License(name="Proprietary"),
    ),
    public=True,
    permission_classes=(permissions.AllowAny,),
)

urlpatterns = [
    path('admin/', admin.site.urls),

    # Login para Swagger / DRF
    path('api-auth/', include('rest_framework.urls', namespace='rest_framework')),

    # 👇 ahora todo lo de biometría vive aquí
    path('api/', include(('biometria.urls', 'biometria'), namespace='biometria')),

    # Swagger / ReDoc
    path('swagger.json', schema_view.without_ui(cache_timeout=0), name='schema-json'),
    path('swagger/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    path('redoc/', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
]
