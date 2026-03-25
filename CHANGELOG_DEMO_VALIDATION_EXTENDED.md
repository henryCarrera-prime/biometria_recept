# Changelog - Demo Validation Extended

## Descripción General

Se ha implementado un nuevo endpoint `/api/biometrics/demo_validation_extended` que extiende la funcionalidad de validación biométrica permitiendo una segunda validación opcional con imagen de registro civil.

## Cambios Realizados

### 1. **Servicio de Aplicación** (`biometria/application/demo_validation_service.py`)

#### Nueva Clase: `DemoValidationExtendedService`
- **Propósito**: Realizar validaciones biométricas con soporte para Registro Civil opcional
- **Método**: `execute(uuid_proceso, cedula_frontal_b64, rostro_persona_b64, registro_civil_b64=None)`
- **Características**:
  - Validación de cédula-rostro (como antes)
  - Validación opcional de registro_civil-rostro
  - Ambas comparaciones son necesarias si se proporciona registro civil
  - Detección de liveness en rostro de persona
  - Umbrales de similitud: ≥95% para ambas comparaciones
  - Logging detallado y diagnósticos
  - Compatibilidad total con el servicio original

#### Flujo de Validación
```
1. Decodificar imágenes (cédula, rostro, registro civil opcional)
2. Detectar rostros en cada imagen
3. Verificar liveness en rostro de persona
4. Comparar similitud: cédula ↔ rostro
5. Comparar similitud: registro_civil ↔ rostro (si se proporciona)
6. Evaluar resultado final
```

---

### 2. **Schemas** (`biometria/presentation/schemas.py`)

#### Nuevos Serializers

**`DemoValidationExtendedRequestSerializer`**
- `uuidProceso` (UUID, requerido): Identificador del proceso
- `cedulaFrontalBase64` (string, requerido): Imagen de cédula en base64
- `rostroPersonaBase64` (string, requerido): Imagen de rostro en base64
- `registroCivilBase64` (string, opcional): Imagen de registro civil en base64

**`DemoValidationExtendedResponseItemSerializer`**
- Incluye todos los campos del response original más:
  - `cedula_rostro_match`: Resultado de comparación cédula-rostro
  - `cedula_rostro_score`: Score de similitud cédula-rostro (0-100%)
  - `registro_civil_rostro_match`: Resultado de comparación registro-rostro (null si no proporcionado)
  - `registro_civil_rostro_score`: Score de similitud registro-rostro (null si no proporcionado)
  - `registro_civil_provided`: Flag indicando si se proporcionó registro civil

**`DemoValidationExtendedResponseSerializer`**
- Estructura completa de respuesta con diagnostics opcionales

---

### 3. **API** (`biometria/presentation/api.py`)

#### Nuevo Endpoint: `DemoValidationExtendedAPIView`

**Ruta**: `POST /api/biometrics/demo_validation_extended`

**Funcionalidades**:
- Validación de parámetros requeridos y opcionales
- Validación de formato base64
- Inicialización de servicios (RekognitionFaceDetector, LuxandClient, RekognitionMatcher)
- Registra flujos en auditoría (`biometria_flows/`)
- Mantiene índice por proceso (`trace_<uuid_proceso>.json`)
- Manejo robusto de errores
- Documentación Swagger integrada

**Comportamiento**:
- Sin `registroCivilBase64`: Realiza validación estándar (igual a demo_validation_service)
- Con `registroCivilBase64`: Realiza ambas validaciones (ampliado)

**Respuesta de Éxito** (status 200):
```json
{
  "status": boolean,
  "message": "String descriptivo",
  "uuidProceso": "UUID",
  "data": [{...}],
  "diagnostics": {...}
}
```

**Respuesta de Error**:
- 400: Parámetros requeridos faltantes
- 422: Formato base64 inválido
- 500: Error no controlado

---

### 4. **URLs** (`biometria/urls.py`)

#### Nueva Ruta Registrada

```python
path('biometrics/demo_validation_extended', 
     DemoValidationExtendedAPIView.as_view(), 
     name='demo-validation-extended')
```

---

## Archivos de Documentación

Se han creado dos archivos de documentación:

### 1. **DEMO_VALIDATION_EXTENDED_DOCS.md**
Documentación técnica completa que incluye:
- Descripción general y diferencias con el servicio anterior
- Parámetros de entrada y salida
- Umbrales y criterios de evaluación
- Ejemplos de respuestas (exitosa, fallida, sin registro)
- Flujo de validación detallado
- Consideraciones de diseño
- Migración desde demo_validation_service
- FAQ y troubleshooting
- Información de auditoría y trazabilidad

### 2. **DEMO_VALIDATION_EXTENDED_EXAMPLES.md**
Ejemplos prácticos de uso:
- Ejemplos con cURL
- Ejemplos con Python (básico, avanzado, con cliente reutilizable)
- Ejemplos con JavaScript/Node.js
- Ejemplo con formulario React
- Pruebas rápidas
- Troubleshooting

---

## Compatibilidad y Migración

### ✅ Compatibilidad Total

El nuevo endpoint es **completamente compatible** con el anterior:

```
demo_validation_service     demo_validation_extended
        (antiguo)                    (nuevo)
             ↓                            ↓
Cédula ↔ Rostro          Cédula ↔ Rostro (SIEMPRE)
                         + Registro ↔ Rostro (OPCIONAL)
```

**Para migrar sin cambios**: simplemente cambiar la URL de `/demo_validation_service` a `/demo_validation_extended`

**Para usar nueva funcionalidad**: agregar el parámetro `registroCivilBase64` opcional

---

## Comportamiento Detallado

### Sin Registro Civil (`registroCivilBase64` omitido)

```
Validaciones:
✓ Detección de liveness
✓ Comparación cédula-rostro (≥95%)

Resultado = (Liveness OK) AND (Cédula-Rostro OK)
```

### Con Registro Civil (`registroCivilBase64` proporcionado)

```
Validaciones:
✓ Detección de liveness
✓ Comparación cédula-rostro (≥95%)
✓ Comparación registro_civil-rostro (≥95%)
✓ Rostros detectados en todas las imágenes

Resultado = (Liveness OK) AND (Cédula-Rostro OK) AND (Registro-Rostro OK)
```

---

## Umbrales y Configuración

| Parámetro | Valor | Descripción |
|-----------|-------|------------|
| Similitud (cédula-rostro) | ≥95% | Umbral de Rekognition |
| Similitud (registro-rostro) | ≥95% | Umbral de Rekognition |
| Liveness | >0.5 | Puntuación normalizada |
| Cédula válida | 100% | Siempre asume válida |

---

## Auditoría y Logs

### Archivos Generados

**Por ejecución**: `biometria_flows/<uuid_validation>.txt`
```json
{
  "type": "demo_validation_extended",
  "uuid_validation": "...",
  "uuid_proceso": "...",
  "started_at_utc": "...",
  "finished_at_utc": "...",
  "registro_civil_provided": true,
  "result": {...},
  "diagnostics": {...}
}
```

**Índice por proceso**: `biometria_flows/trace_<uuid_proceso>.json`
- Mantiene todas las ejecuciones de un proceso
- Actualizado automáticamente
- Usado para trazabilidad

---

## Testing

### Prueba Manual con cURL

```bash
# Sin registro civil
curl -X POST http://localhost:8000/api/biometrics/demo_validation_extended \
  -H "Content-Type: application/json" \
  -d '{
    "uuidProceso": "04205d9c-1439-4a6e-a06c-21012a4ea744",
    "cedulaFrontalBase64": "data:image/jpeg;base64,...",
    "rostroPersonaBase64": "data:image/jpeg;base64,..."
  }'

# Con registro civil
curl -X POST http://localhost:8000/api/biometrics/demo_validation_extended \
  -H "Content-Type: application/json" \
  -d '{
    "uuidProceso": "04205d9c-1439-4a6e-a06c-21012a4ea744",
    "cedulaFrontalBase64": "data:image/jpeg;base64,...",
    "rostroPersonaBase64": "data:image/jpeg;base64,...",
    "registroCivilBase64": "data:image/jpeg;base64,..."
  }'
```

Ver `DEMO_VALIDATION_EXTENDED_EXAMPLES.md` para más ejemplos.

---

## Cambios en el Código

### Archivos Modificados

1. **`biometria/application/demo_validation_service.py`**
   - Agregada clase `DemoValidationExtendedService` (~600 líneas)
   - Mantiene clase `DemoValidationService` sin cambios

2. **`biometria/presentation/schemas.py`**
   - Agregado `DemoValidationExtendedRequestSerializer`
   - Agregado `DemoValidationExtendedResponseItemSerializer`
   - Agregado `DemoValidationExtendedResponseSerializer`
   - Serializers originales sin cambios

3. **`biometria/presentation/api.py`**
   - Importada `DemoValidationExtendedService`
   - Importados nuevos serializers
   - Agregada clase `DemoValidationExtendedAPIView` (~250 líneas)
   - Endpoint original sin cambios

4. **`biometria/urls.py`**
   - Importado `DemoValidationExtendedAPIView`
   - Registrada nueva ruta
   - Rutas originales sin cambios

### Archivos Nuevos

1. **`DEMO_VALIDATION_EXTENDED_DOCS.md`**
   - Documentación técnica completa
   - ~500 líneas de contenido

2. **`DEMO_VALIDATION_EXTENDED_EXAMPLES.md`**
   - Ejemplos prácticos de uso
   - ~700 líneas de código y documentación

---

## Consideraciones de Rendimiento

- **Tiempo de procesamiento**: 5-15 segundos (esperar timeout adecuado)
- **Tamaño de payload**: Base64 puede ser voluminoso (~1-2MB por imagen)
- **Llamadas a servicios externos**: 
  - Rekognition (2 llamadas si registro civil se proporciona)
  - Luxand (1 llamada para liveness)
- **Almacenamiento**: Un archivo TXT por ejecución (pequeño, solo metadatos)

---

## Backward Compatibility

✅ **100% Compatible**

- Endpoint original `/demo_validation_service` sigue funcionando sin cambios
- Nuevo endpoint es una extensión, no un reemplazo
- Estructura de respuesta es compatible (campos adicionales son opcionales)
- Clientes existentes pueden migrar gradualmente

---

## Próximos Pasos Recomendados

1. **Pruebas manual**: Usar ejemplos en `DEMO_VALIDATION_EXTENDED_EXAMPLES.md`
2. **Pruebas de integración**: Validar con datos reales
3. **Monitoreo**: Revisar logs en `biometria_flows/`
4. **Feedback**: Ajustar umbrales si es necesario
5. **Documentación interna**: Compartir con equipo de desarrollo

---

## Resumen

Se ha implementado exitosamente un nuevo endpoint de validación biométrica extendida que:

✅ Permite validación con registro civil opcional
✅ Mantiene total compatibilidad con el servicio anterior
✅ Proporciona espectro de validación más amplio
✅ Incluye documentación completa y ejemplos
✅ Mantiene auditoría y trazabilidad
✅ Maneja errores robustamente
✅ Incluye diagnósticos detallados

**Endpoint disponible en**: `POST /api/biometrics/demo_validation_extended`

