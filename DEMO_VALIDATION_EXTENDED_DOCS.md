# Demo Validation Extended Service - Documentación

## Descripción General

El endpoints `demo_validation_extended` es una extensión del servicio `demo_validation_service` que permite validaciones biométricas más amplias mediante la incorporación opcional de una segunda imagen de comparación (imagen de registro civil).

### Diferencias clave

| Aspecto | demo_validation_service | demo_validation_extended |
|--------|------------------------|--------------------------|
| Parámetro Registro Civil | ❌ No soportado | ✅ Soportado (opcional) |
| Comparación principal | Cédula ↔ Rostro | Cédula ↔ Rostro |
| Comparación secundaria | - | Registro Civil ↔ Rostro (opcional) |
| Espectro de validación | Estándar | Más amplio (si se proporciona segundo documento) |
| Compatibilidad inversa | - | ✅ Totalmente compatible (sin parámetro = igual comportamiento) |

## Endpoint

```
POST /api/biometrics/demo_validation_extended
```

## Parámetros de Entrada

### Requeridos

| Campo | Tipo | Descripción |
|-------|------|------------|
| `uuidProceso` | UUID | Identificador único del proceso general |
| `cedulaFrontalBase64` | String (base64) | Imagen de la cédula frontal en formato base64 con prefijo MIME |
| `rostroPersonaBase64` | String (base64) | Imagen del rostro de la persona en formato base64 con prefijo MIME |

### Opcionales

| Campo | Tipo | Descripción | Comportamiento si no se proporciona |
|-------|------|------------|-------------------------------------|
| `registroCivilBase64` | String (base64) | Imagen del registro civil en formato base64 con prefijo MIME | Se realiza solo comparación cédula-rostro (igual a `demo_validation_service`) |

## Ejemplo de Uso

### Caso 1: Sin Registro Civil (comportamiento estándar)

```json
POST /api/biometrics/demo_validation_extended

{
  "uuidProceso": "04205d9c-1439-4a6e-a06c-21012a4ea744",
  "cedulaFrontalBase64": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "rostroPersonaBase64": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

**Validaciones realizadas:**
1. Liveness en rostro de persona ✅
2. Comparación: Cédula ↔ Rostro ✅

---

### Caso 2: Con Registro Civil (validación ampliada)

```json
POST /api/biometrics/demo_validation_extended

{
  "uuidProceso": "04205d9c-1439-4a6e-a06c-21012a4ea744",
  "cedulaFrontalBase64": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "rostroPersonaBase64": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "registroCivilBase64": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

**Validaciones realizadas:**
1. Liveness en rostro de persona ✅
2. Comparación: Cédula ↔ Rostro ✅
3. Comparación: Registro Civil ↔ Rostro ✅

**Criterio de éxito:** Los tres requisitos anterior deben cumplirse para una validación exitosa.

---

## Respuesta

### Estructura General

```json
{
  "status": boolean,
  "message": "String descriptivo del resultado",
  "uuidProceso": "UUID del proceso",
  "data": [
    {
      "uuid_validation": "UUID de esta validación",
      "evaluacion": 85.5,
      "cedula_valida": true,
      "liveness_detectado": true,
      "rostros_coinciden": true,
      "score_cedula": 100.0,
      "score_liveness": 87.5,
      "score_similarity": 96.8,
      "cedula_rostro_match": true,
      "cedula_rostro_score": 96.8,
      "registro_civil_rostro_match": true,        // null si no se proporcionó
      "registro_civil_rostro_score": 92.5,        // null si no se proporcionó
      "registro_civil_provided": true,             // false si no se proporcionó
    }
  ],
  "diagnostics": {
    // Información detallada para debugging
    "cedula_image_size": "640x480",
    "rostro_image_size": "800x600",
    "registro_civil_image_size": "600x480",
    "has_registro_civil": true,
    ...
  }
}
```

### Campos de Respuesta

#### Validación Global

- **status** (boolean): `true` si TODAS las validaciones pasaron, `false` en caso contrario
- **message** (string): Descripción textual del resultado
- **uuidProceso** (UUID): Identificador del proceso

#### Scores y Resultados

| Campo | Descripción | Rango | Notas |
|-------|------------|-------|-------|
| `evaluacion` | Score promedio de evaluación | 0-100 | Promedio de cédula, liveness y similitud |
| `score_cedula` | Score de validación de cédula | 0-100 | Siempre 100 (asume válida) |
| `score_liveness` | Score de detección de vida | 0-100 | Basado en análisis pasivo de movimiento |
| `score_similarity` | Score de similitud promedio | 0-100 | Promedio de ambas comparaciones si aplica |
| `cedula_rostro_score` | Score cédula-rostro | 0-100 | Porcentaje de similitud de Rekognition |
| `registro_civil_rostro_score` | Score registro-rostro | 0-100 o null | null si no se proporcionó |

#### Flags de Coincidencia

- **cedula_rostro_match** (boolean): ¿Cédula y rostro coinciden? (threshold: ≥95%)
- **registro_civil_rostro_match** (boolean o null): ¿Registro civil y rostro coinciden? (threshold: ≥95%, null si no proporcionado)
- **liveness_detectado** (boolean): ¿Se detectó liveness? (threshold: >0.5)
- **rostros_coinciden** (boolean): ¿Todas las comparaciones de rostro coinciden?

#### Información de Disponibilidad

- **registro_civil_provided** (boolean): Indica si se proporcionó la imagen de registro civil
- **cedula_valida** (boolean): Siempre true (validación de cédula deshabilitada)

---

## Códigos de Respuesta HTTP

| Código | Significado | Cuando Ocurre |
|--------|------------|---------------|
| 200 | OK | Siempre (incluso con validaciones fallidas) |
| 400 | Bad Request | Faltan parámetros requeridos |
| 422 | Unprocessable Entity | Formato base64 inválido |
| 500 | Internal Server Error | Error no controlado durante procesamiento |

---

## Lujo de Validación Detallado

```
1. DECODIFICACIÓN
   ├── Decodificar cedulaFrontalBase64
   ├── Decodificar rostroPersonaBase64
   └── Decodificar registroCivilBase64 (si se proporciona)

2. DETECCIÓN DE ROSTROS
   ├── Detectar rostro en cédula
   ├── Detectar rostro en rostro de persona
   └── Detectar rostro en registro civil (si se proporciona)

3. LIVENESS
   └── Verificar señales de vida en rostro de persona

4. SIMILITUD - CÉDULA-ROSTRO
   └── Comparar rostro de cédula con rostro de persona

5. SIMILITUD - REGISTRO CIVIL-ROSTRO (opcional)
   └── Comparar rostro de registro civil con rostro de persona (si se proporciona)

6. EVALUACIÓN FINAL
   ├── Si registro civil NO se proporcionó:
   │   └── Resultado = (Cédula OK) AND (Rostros coinciden) AND (Liveness OK)
   └── Si registro civil SÍ se proporcionó:
       └── Resultado = (Cédula OK) AND (Cédula-Rostro OK) AND (Registro-Rostro OK) AND (Liveness OK)
```

---

## Consideraciones de Diseño

### Compatibilidad Inversa
El endpoint es **totalmente compatible** con el flujo del `demo_validation_service`:
- Si no se proporciona `registroCivilBase64`, realiza exactamente las mismas validaciones
- Clientes existentes pueden migrar sin cambios en su lógica

### Umbrales de Similitud
- **Cédula-Rostro**: ≥ 95% (Rekognition)
- **Registro Civil-Rostro**: ≥ 95% (Rekognition)
- **Liveness**: > 0.5 (puntuación normalizada)

### Manejo de Errores
- Si la decodificación de `registroCivilBase64` falla, se ignora y se procede sin ella
- Si el reconocimiento facial falla en cualquier imagen, se intenta usar la imagen completa
- Los diagnósticos incluyen información detallada para debugging

### Almacenamiento
Todas las ejecuciones se registran en `biometria_flows/` con:
- Archivo individual: `<uuid_validation>.txt`
- Índice por proceso: `trace_<uuid_proceso>.json`

---

## Ejemplos de Respuesta

### Validación Exitosa (con Registro Civil)

```json
{
  "status": true,
  "message": "Validación exitosa: cédula válida, liveness detectado, cedula-rostro coinciden y registro_civil-rostro coinciden.",
  "uuidProceso": "04205d9c-1439-4a6e-a06c-21012a4ea744",
  "data": [
    {
      "uuid_validation": "a1b2c3d4-e5f6-47g8-h9i0-j1k2l3m4n5o6",
      "evaluacion": 93.27,
      "cedula_valida": true,
      "liveness_detectado": true,
      "rostros_coinciden": true,
      "score_cedula": 100.0,
      "score_liveness": 92.05,
      "score_similarity": 94.65,
      "cedula_rostro_match": true,
      "cedula_rostro_score": 96.8,
      "registro_civil_rostro_match": true,
      "registro_civil_rostro_score": 92.5,
      "registro_civil_provided": true
    }
  ]
}
```

### Validación Fallida (Registro Civil no coincide)

```json
{
  "status": false,
  "message": "Validación fallida: registro_civil-rostro no coinciden.",
  "uuidProceso": "04205d9c-1439-4a6e-a06c-21012a4ea744",
  "data": [
    {
      "uuid_validation": "b2c3d4e5-f6g7-48h9-i0j1-k2l3m4n5o6p7",
      "evaluacion": 82.43,
      "cedula_valida": true,
      "liveness_detectado": true,
      "rostros_coinciden": false,
      "score_cedula": 100.0,
      "score_liveness": 88.3,
      "score_similarity": 82.6,
      "cedula_rostro_match": true,
      "cedula_rostro_score": 96.8,
      "registro_civil_rostro_match": false,
      "registro_civil_rostro_score": 68.4,
      "registro_civil_provided": true
    }
  ]
}
```

### Validación sin Registro Civil (Compatible)

```json
{
  "status": true,
  "message": "Validación exitosa: cédula válida, liveness detectado y rostros coinciden.",
  "uuidProceso": "04205d9c-1439-4a6e-a06c-21012a4ea744",
  "data": [
    {
      "uuid_validation": "c3d4e5f6-g7h8-49i9-j0k1-l2m3n4o5p6q7",
      "evaluacion": 94.6,
      "cedula_valida": true,
      "liveness_detectado": true,
      "rostros_coinciden": true,
      "score_cedula": 100.0,
      "score_liveness": 91.5,
      "score_similarity": 92.2,
      "cedula_rostro_match": true,
      "cedula_rostro_score": 92.2,
      "registro_civil_rostro_match": null,
      "registro_civil_rostro_score": null,
      "registro_civil_provided": false
    }
  ]
}
```

---

## Formato de Imágenes Base64

Las imágenes deben incluir el prefijo MIME:

```
data:image/jpeg;base64,<contenido_base64>
o
data:image/png;base64,<contenido_base64>
```

### Ejemplo de Generación en Python

```python
import base64

with open("imagen.jpg", "rb") as f:
    img_data = base64.b64encode(f.read()).decode()
    b64_image = f"data:image/jpeg;base64,{img_data}"
```

### Ejemplo de Generación en JavaScript

```javascript
const file = document.getElementById('imageInput').files[0];
const reader = new FileReader();
reader.onload = (e) => {
    const b64_image = e.target.result; // Ya incluye el prefijo
    // Enviar en el body de la solicitud
};
reader.readAsDataURL(file);
```

---

## Auditoría y Trazabilidad

### Archivo de Flujo Individual

Cada validación genera un archivo en `biometria_flows/<uuid_validation>.txt`:

```json
{
  "type": "demo_validation_extended",
  "uuid_validation": "...",
  "uuid_proceso": "...",
  "started_at_utc": "2024-03-25T10:30:45.123456+00:00",
  "finished_at_utc": "2024-03-25T10:30:47.654321+00:00",
  "registro_civil_provided": true,
  "result": { ... },
  "diagnostics": { ... }
}
```

### Índice de Proceso

Se mantiene un índice en `biometria_flows/trace_<uuid_proceso>.json` con todas las ejecuciones del proceso.

---

## Migración desde demo_validation_service

Para migrar del endpoint antiguo:

1. **Sin cambios**: Si no usas Registro Civil
   ```
   Cambiar: POST /biometrics/demo_validation_service
   Por:     POST /biometrics/demo_validation_extended
   (El comportamiento es idéntico)
   ```

2. **Con Registro Civil** (nuevo):
   ```json
   {
     "uuidProceso": "...",
     "cedulaFrontalBase64": "...",
     "rostroPersonaBase64": "...",
     "registroCivilBase64": "..."  // Nuevo parámetro
   }
   ```

---

## Soporte y Debugging

### Diagnósticos Disponibles

El campo `diagnostics` en la respuesta contiene:
- Tamaños de imagen
- Detección de rostros (bounding boxes)
- Scores de liveness
- Scores de similitud por comparación
- Errores detallados si aplica

### Logging

El servicio registra en `biometria.verify`:

```python
logger.info("✅ Rostro detectado en cédula")
logger.warning("⚠️ No se detectó rostro en cédula localmente")
logger.error("❌ Error en detección de rostro: {error}")
```

---

## Preguntas Frecuentes

**P: ¿Qué pasa si proporciono un Registro Civil de mala calidad?**
R: Si el rostro no se detecta, la validación fallará. Se recomienda usar imágenes de buena claridad.

**P: ¿Puedo usar el endpoint sin Registro Civil?**
R: Sí, completamente. Se realiza la validación estándar (cédula-rostro).

**P: ¿Cuál es el umbral de similitud?**
R: 95% tanto para cédula-rostro como para registro_civil-rostro.

**P: ¿Se guardan las imágenes?**
R: No, solo se guardan metadatos y scores en los logs.

**P: ¿Puedo reutilizar una validación anterior?**
R: No, cada llamada genera un `uuid_validation` nuevo y se registra como una ejecución independiente.

---

## Historial de Cambios

### v1.0 (2024-03-25)
- Lanzamiento inicial de `demo_validation_extended`
- Soporte para Registro Civil opcional
- Compatibilidad total con `demo_validation_service`
- Documentación completa

