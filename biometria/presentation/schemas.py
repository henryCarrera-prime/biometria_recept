# biometria/presentation/schemas.py
from rest_framework import serializers

# ---------- Verify (POST) ----------
class BiometricsVerifyRequestSerializer(serializers.Serializer):
    uuidProceso = serializers.UUIDField(help_text="UUID del proceso general.")
    referenceImageUrl = serializers.CharField(help_text="Ruta local o S3 (prefix o archivo) de la imagen de referencia.")
    framesIndexUrl = serializers.CharField(help_text="Ruta local o S3 (prefix o archivo) de los frames a evaluar.")

class BiometricsVerifyResponseItemSerializer(serializers.Serializer):
    uuid_proceso_biometria = serializers.UUIDField()
    evaluacion = serializers.FloatField()

class BiometricsVerifyResponseSerializer(serializers.Serializer):
    status = serializers.CharField()
    message = serializers.CharField()
    data = BiometricsVerifyResponseItemSerializer(many=True)

# ---------- Flow (.txt -> JSON) ----------
class ThresholdsSerializer(serializers.Serializer):
    similarity = serializers.FloatField()
    live = serializers.FloatField()
    luxand = serializers.FloatField()

class ResultSerializer(serializers.Serializer):
    status = serializers.CharField()
    message = serializers.CharField()
    is_live = serializers.BooleanField()
    is_match = serializers.BooleanField()
    liveness_score = serializers.FloatField()
    similarity = serializers.FloatField()
    evaluation_pct = serializers.FloatField()

class FlowLinksSerializer(serializers.Serializer):
    self = serializers.CharField()
    download = serializers.CharField()

class BiometricsFlowSerializer(serializers.Serializer):
    uuid_proceso_biometria = serializers.UUIDField()
    uuid_proceso = serializers.UUIDField()
    started_at_utc = serializers.CharField()
    finished_at_utc = serializers.CharField()
    reference_uri = serializers.CharField()
    frames_uri = serializers.CharField()
    frames_count = serializers.IntegerField()
    thresholds = ThresholdsSerializer()
    result = ResultSerializer()
    _links = FlowLinksSerializer(required=False)

# ---------- Trace (GET por uuidProceso) ----------
class PageMetaSerializer(serializers.Serializer):
    offset = serializers.IntegerField()
    limit = serializers.IntegerField()
    returned = serializers.IntegerField()
    total = serializers.IntegerField()
    has_more = serializers.BooleanField()

class TraceItemSerializer(serializers.Serializer):
    uuid_proceso_biometria = serializers.UUIDField()
    status = serializers.CharField()
    message = serializers.CharField()
    evaluation_pct = serializers.FloatField(allow_null=True)
    similarity = serializers.FloatField(allow_null=True)
    is_live = serializers.BooleanField()
    is_match = serializers.BooleanField()
    liveness_score = serializers.FloatField(allow_null=True)
    reference_uri = serializers.CharField()
    frames_uri = serializers.CharField()
    frames_count = serializers.IntegerField()
    started_at_utc = serializers.CharField()
    finished_at_utc = serializers.CharField()

class TraceResponseSerializer(serializers.Serializer):
    uuid_proceso = serializers.UUIDField()
    count = serializers.IntegerField()
    items = TraceItemSerializer(many=True)
    page = PageMetaSerializer()

# ---------- Demo Validation (POST) ----------
class DemoValidationRequestSerializer(serializers.Serializer):
    uuidProceso = serializers.UUIDField(help_text="UUID del proceso general.")
    cedulaFrontalBase64 = serializers.CharField(help_text="Imagen de la cédula frontal en base64.")
    rostroPersonaBase64 = serializers.CharField(help_text="Imagen del rostro de la persona en base64.")

class DemoValidationResponseItemSerializer(serializers.Serializer):
    uuid_validation = serializers.UUIDField()
    evaluacion = serializers.FloatField()
    cedula_valida = serializers.BooleanField()
    liveness_detectado = serializers.BooleanField()
    rostros_coinciden = serializers.BooleanField()
    score_cedula = serializers.FloatField()
    score_liveness = serializers.FloatField()
    score_similarity = serializers.FloatField()

class DemoValidationResponseSerializer(serializers.Serializer):
    status = serializers.BooleanField()
    message = serializers.CharField()
    uuidProceso = serializers.UUIDField()
    data = DemoValidationResponseItemSerializer(many=True)

# ---------- Demo Validation Extended (POST) - Con soporte para registro civil ----------
class DemoValidationExtendedRequestSerializer(serializers.Serializer):
    uuidProceso = serializers.UUIDField(help_text="UUID del proceso general.")
    cedulaFrontalBase64 = serializers.CharField(help_text="Imagen de la cédula frontal en base64.")
    rostroPersonaBase64 = serializers.CharField(help_text="Imagen del rostro de la persona en base64.")
    registroCivilBase64 = serializers.CharField(
        required=False,
        allow_blank=True,
        help_text="(OPCIONAL) Imagen del registro civil en base64. Si no se proporciona, solo se compara cedula-rostro."
    )

class DemoValidationExtendedResponseItemSerializer(serializers.Serializer):
    uuid_validation = serializers.UUIDField()
    evaluacion = serializers.FloatField()
    cedula_valida = serializers.BooleanField()
    liveness_detectado = serializers.BooleanField()
    rostros_coinciden = serializers.BooleanField()
    score_cedula = serializers.FloatField()
    score_liveness = serializers.FloatField()
    score_similarity = serializers.FloatField()
    cedula_rostro_match = serializers.BooleanField()
    cedula_rostro_score = serializers.FloatField()
    registro_civil_rostro_match = serializers.BooleanField(allow_null=True)
    registro_civil_rostro_score = serializers.FloatField(allow_null=True)
    registro_civil_provided = serializers.BooleanField()

class DemoValidationExtendedResponseSerializer(serializers.Serializer):
    status = serializers.BooleanField()
    message = serializers.CharField()
    uuidProceso = serializers.UUIDField()
    data = DemoValidationExtendedResponseItemSerializer(many=True)
    diagnostics = serializers.JSONField(required=False)
