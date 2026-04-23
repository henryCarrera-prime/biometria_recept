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
    cedulaFrontalBase64 = serializers.CharField(
        required=False,
        allow_blank=True,
        help_text="(OPCIONAL) Imagen de la c?dula frontal en base64."
    )
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
    cedula_rostro_match = serializers.BooleanField(allow_null=True)
    cedula_rostro_score = serializers.FloatField(allow_null=True)
    registro_civil_rostro_match = serializers.BooleanField(allow_null=True)
    registro_civil_rostro_score = serializers.FloatField(allow_null=True)
    registro_civil_provided = serializers.BooleanField()
    cedula_provided = serializers.BooleanField(required=False)
    cedula_validacion_aplica = serializers.BooleanField(required=False)
    cantidad_imagenes_comparadas = serializers.IntegerField(required=False)
    porcentaje_global_similitud = serializers.FloatField(required=False)
    porcentaje_vida = serializers.FloatField(required=False)
    timestamp_ecuador = serializers.CharField(required=False)
    identificador_proceso_asignado = serializers.CharField(required=False)
    pares_comparados = serializers.JSONField(required=False)
    imagenes_comparadas = serializers.JSONField(required=False)
    puntos_evaluados = serializers.JSONField(required=False)


class DemoValidationExtendedDiagnosticsThresholdsSerializer(serializers.Serializer):
    liveness_threshold = serializers.FloatField(required=False)
    similarity_threshold = serializers.FloatField(required=False)
    cedula_threshold = serializers.FloatField(required=False)
    cedula_positive_labels = serializers.ListField(
        child=serializers.IntegerField(),
        required=False
    )


class DemoValidationExtendedDiagnosticsInputsSerializer(serializers.Serializer):
    cedula_provided = serializers.BooleanField(required=False)
    rostro_provided = serializers.BooleanField(required=False)
    registro_civil_provided = serializers.BooleanField(required=False)


class DemoValidationExtendedDiagnosticsFaceItemSerializer(serializers.Serializer):
    face_found = serializers.BooleanField(allow_null=True, required=False)
    bbox = serializers.JSONField(required=False, allow_null=True)
    landmarks_count = serializers.IntegerField(required=False, allow_null=True)


class DemoValidationExtendedDiagnosticsFaceDetectionSerializer(serializers.Serializer):
    cedula = DemoValidationExtendedDiagnosticsFaceItemSerializer(required=False)
    rostro = DemoValidationExtendedDiagnosticsFaceItemSerializer(required=False)
    registro_civil = DemoValidationExtendedDiagnosticsFaceItemSerializer(required=False)


class DemoValidationExtendedDiagnosticsLivenessSerializer(serializers.Serializer):
    is_live = serializers.BooleanField(required=False)
    score = serializers.FloatField(required=False)
    score_pct = serializers.FloatField(required=False)
    threshold = serializers.FloatField(required=False)


class DemoValidationExtendedDiagnosticsSimilaritySerializer(serializers.Serializer):
    pairs = serializers.JSONField(required=False)
    global_score_pct = serializers.FloatField(required=False)
    all_pairs_match = serializers.BooleanField(required=False)
    pairs_count = serializers.IntegerField(required=False)


class DemoValidationExtendedDiagnosticsSerializer(serializers.Serializer):
    uuid_validation = serializers.CharField(required=False)
    identificador_proceso_asignado = serializers.CharField(required=False)
    timestamp_ecuador = serializers.CharField(required=False)
    parametrizacion = DemoValidationExtendedDiagnosticsThresholdsSerializer(required=False)
    inputs = DemoValidationExtendedDiagnosticsInputsSerializer(required=False)
    face_detection = DemoValidationExtendedDiagnosticsFaceDetectionSerializer(required=False)
    liveness = DemoValidationExtendedDiagnosticsLivenessSerializer(required=False)
    similarity = DemoValidationExtendedDiagnosticsSimilaritySerializer(required=False)
    errors = serializers.ListField(child=serializers.CharField(), required=False)
    error = serializers.CharField(required=False)

class DemoValidationExtendedResponseSerializer(serializers.Serializer):
    status = serializers.BooleanField()
    message = serializers.CharField()
    uuidProceso = serializers.UUIDField()
    data = DemoValidationExtendedResponseItemSerializer(many=True)
    diagnostics = DemoValidationExtendedDiagnosticsSerializer(required=False)
