# Ejemplos de Uso - Demo Validation Extended

Este archivo contiene ejemplos prácticos para usar el nuevo endpoint `/api/biometrics/demo_validation_extended`.

## Tabla de Contenidos

- [Requisitos](#requisitos)
- [Ejemplos con cURL](#ejemplos-con-curl)
- [Ejemplos con Python](#ejemplos-con-python)
- [Ejemplos con JavaScript/Node.js](#ejemplos-con-javascriptnodejs)

---

## Requisitos

- URL base del API: `http://localhost:8000` (o tu servidor)
- Imágenes en formato JPEG o PNG
- Las imágenes deben ser legibles (rostros claros)

---

## Ejemplos con cURL

### 1. Validación SIN Registro Civil (comportamiento estándar)

```bash
curl -X POST http://localhost:8000/api/biometrics/demo_validation_extended \
  -H "Content-Type: application/json" \
  -d '{
    "uuidProceso": "04205d9c-1439-4a6e-a06c-21012a4ea744",
    "cedulaFrontalBase64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQA...",
    "rostroPersonaBase64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQA..."
  }'
```

### 2. Validación CON Registro Civil (validación ampliada)

```bash
curl -X POST http://localhost:8000/api/biometrics/demo_validation_extended \
  -H "Content-Type: application/json" \
  -d '{
    "uuidProceso": "04205d9c-1439-4a6e-a06c-21012a4ea744",
    "cedulaFrontalBase64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQA...",
    "rostroPersonaBase64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQA...",
    "registroCivilBase64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQA..."
  }'
```

### 3. Generar base64 desde archivo y enviar (Linux/Mac)

```bash
# Función auxiliar para convertir a base64
encode_image() {
    local file=$1
    local mime_type=$(file -b --mime-type "$file")
    echo "data:${mime_type};base64,$(base64 -i "$file")"
}

# Usar la función
CEDULA_B64=$(encode_image cedula.jpg)
ROSTRO_B64=$(encode_image rostro.jpg)
REGISTRO_B64=$(encode_image registro_civil.jpg)

curl -X POST http://localhost:8000/api/biometrics/demo_validation_extended \
  -H "Content-Type: application/json" \
  -d "{
    \"uuidProceso\": \"04205d9c-1439-4a6e-a06c-21012a4ea744\",
    \"cedulaFrontalBase64\": \"$CEDULA_B64\",
    \"rostroPersonaBase64\": \"$ROSTRO_B64\",
    \"registroCivilBase64\": \"$REGISTRO_B64\"
  }"
```

---

## Ejemplos con Python

### 1. Validación básica sin Registro Civil

```python
import requests
import base64
import uuid
import json

API_URL = "http://localhost:8000/api/biometrics/demo_validation_extended"

def encode_image_to_base64(image_path):
    """Convierte una imagen a base64 con prefijo MIME"""
    import mimetypes
    mime_type, _ = mimetypes.guess_type(image_path)
    with open(image_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode()
    return f"data:{mime_type};base64,{img_data}"

# Preparar imágenes
cedula_b64 = encode_image_to_base64("cedula.jpg")
rostro_b64 = encode_image_to_base64("rostro.jpg")

# Preparar payload
payload = {
    "uuidProceso": str(uuid.uuid4()),
    "cedulaFrontalBase64": cedula_b64,
    "rostroPersonaBase64": rostro_b64
}

# Enviar solicitud
response = requests.post(API_URL, json=payload)
result = response.json()

# Mostrar resultado
print("Status:", result["status"])
print("Message:", result["message"])
print("Data:", json.dumps(result["data"], indent=2))
```

### 2. Validación con Registro Civil

```python
import requests
import base64
import uuid
import json

API_URL = "http://localhost:8000/api/biometrics/demo_validation_extended"

def encode_image_to_base64(image_path):
    import mimetypes
    mime_type, _ = mimetypes.guess_type(image_path)
    with open(image_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode()
    return f"data:{mime_type};base64,{img_data}"

# Preparar imágenes
cedula_b64 = encode_image_to_base64("cedula.jpg")
rostro_b64 = encode_image_to_base64("rostro.jpg")
registro_b64 = encode_image_to_base64("registro_civil.jpg")

# Preparar payload
payload = {
    "uuidProceso": str(uuid.uuid4()),
    "cedulaFrontalBase64": cedula_b64,
    "rostroPersonaBase64": rostro_b64,
    "registroCivilBase64": registro_b64  # Nuevo parámetro
}

# Enviar solicitud
response = requests.post(API_URL, json=payload)
result = response.json()

# Procesar resultado
if result["status"]:
    print("✅ Validación exitosa!")
    data = result["data"][0]
    print(f"   Score promedio: {data['evaluacion']:.2f}%")
    print(f"   Cédula-Rostro: {data['cedula_rostro_score']:.2f}%")
    if data["registro_civil_provided"]:
        print(f"   Registro-Rostro: {data['registro_civil_rostro_score']:.2f}%")
else:
    print("❌ Validación fallida")
    print(f"   Razón: {result['message']}")
```

### 3. Manejo completo con validaciones

```python
import requests
import base64
import uuid
from pathlib import Path

class BiometriaClient:
    def __init__(self, api_url="http://localhost:8000/api/biometrics"):
        self.api_url = api_url
    
    @staticmethod
    def encode_image(image_path):
        """Convierte imagen a base64 con prefijo MIME"""
        import mimetypes
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Imagen no encontrada: {image_path}")
        
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type or not mime_type.startswith("image/"):
            raise ValueError(f"Tipo MIME inválido: {mime_type}")
        
        with open(image_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode()
        return f"data:{mime_type};base64,{img_data}"
    
    def validate_with_registro_civil(self, cedula_path, rostro_path, registro_path=None):
        """
        Valida biometría con o sin registro civil
        
        Args:
            cedula_path: Ruta a imagen de cédula
            rostro_path: Ruta a imagen de rostro
            registro_path: (Opcional) Ruta a imagen de registro civil
        
        Returns:
            dict: Resultado de la validación
        """
        try:
            # Codificar imágenes
            cedula_b64 = self.encode_image(cedula_path)
            rostro_b64 = self.encode_image(rostro_path)
            
            payload = {
                "uuidProceso": str(uuid.uuid4()),
                "cedulaFrontalBase64": cedula_b64,
                "rostroPersonaBase64": rostro_b64
            }
            
            # Agregar registro civil si se proporciona
            if registro_path:
                registro_b64 = self.encode_image(registro_path)
                payload["registroCivilBase64"] = registro_b64
            
            # Enviar solicitud
            response = requests.post(
                f"{self.api_url}/demo_validation_extended",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            return response.json()
        
        except FileNotFoundError as e:
            return {"status": False, "error": str(e)}
        except requests.exceptions.RequestException as e:
            return {"status": False, "error": f"Error de conexión: {e}"}
        except Exception as e:
            return {"status": False, "error": f"Error inesperado: {e}"}
    
    def print_result(self, result):
        """Imprime resultado de forma legible"""
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return
        
        status = result.get("status", False)
        message = result.get("message", "Sin mensaje")
        
        print(f"\nResultado: {'✅ EXITOSO' if status else '❌ FALLIDO'}")
        print(f"Mensaje: {message}")
        
        data = result.get("data", [{}])[0]
        print(f"\nScores:")
        print(f"  Evaluación promedio: {data.get('evaluacion', 0):.2f}%")
        print(f"  Cédula-Rostro:       {data.get('cedula_rostro_score', 0):.2f}%")
        
        if data.get("registro_civil_provided"):
            print(f"  Registro-Rostro:     {data.get('registro_civil_rostro_score', 0):.2f}%")
        
        print(f"\nDetección:")
        print(f"  Liveness:           {'✅' if data.get('liveness_detectado') else '❌'}")
        print(f"  Rostros coinciden:  {'✅' if data.get('rostros_coinciden') else '❌'}")

# Uso
if __name__ == "__main__":
    client = BiometriaClient()
    
    # Validación sin registro civil
    result = client.validate_with_registro_civil("cedula.jpg", "rostro.jpg")
    client.print_result(result)
    
    # Validación con registro civil
    result = client.validate_with_registro_civil(
        "cedula.jpg",
        "rostro.jpg",
        "registro_civil.jpg"
    )
    client.print_result(result)
```

---

## Ejemplos con JavaScript/Node.js

### 1. Validación básica con Node.js

```javascript
const axios = require('axios');
const fs = require('fs');
const path = require('path');
const { v4: uuidv4 } = require('uuid');

const API_URL = 'http://localhost:8000/api/biometrics/demo_validation_extended';

async function encodeImageToBase64(imagePath) {
    const mimeTypes = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png'
    };
    
    const ext = path.extname(imagePath).toLowerCase();
    const mimeType = mimeTypes[ext] || 'image/jpeg';
    
    const imageData = fs.readFileSync(imagePath);
    const base64Data = imageData.toString('base64');
    
    return `data:${mimeType};base64,${base64Data}`;
}

async function validateWithRegistroCivil(cedulaPath, rostroPath, registroPath = null) {
    try {
        // Codificar imágenes
        const cedulaB64 = await encodeImageToBase64(cedulaPath);
        const rostroB64 = await encodeImageToBase64(rostroPath);
        
        const payload = {
            uuidProceso: uuidv4(),
            cedulaFrontalBase64: cedulaB64,
            rostroPersonaBase64: rostroB64
        };
        
        // Agregar registro civil si se proporciona
        if (registroPath) {
            payload.registroCivilBase64 = await encodeImageToBase64(registroPath);
        }
        
        // Enviar solicitud
        const response = await axios.post(API_URL, payload);
        return response.data;
    } catch (error) {
        return {
            status: false,
            error: error.message
        };
    }
}

// Uso
(async () => {
    console.log('Validando con Registro Civil...\n');
    
    const result = await validateWithRegistroCivil(
        'cedula.jpg',
        'rostro.jpg',
        'registro_civil.jpg'
    );
    
    if (result.status) {
        console.log('✅ Validación EXITOSA');
    } else {
        console.log('❌ Validación FALLIDA');
    }
    
    console.log(`Mensaje: ${result.message}`);
    
    const data = result.data?.[0];
    if (data) {
        console.log(`\nScores:`);
        console.log(`  Evaluación: ${data.evaluacion?.toFixed(2)}%`);
        console.log(`  Cédula-Rostro: ${data.cedula_rostro_score?.toFixed(2)}%`);
        if (data.registro_civil_provided) {
            console.log(`  Registro-Rostro: ${data.registro_civil_rostro_score?.toFixed(2)}%`);
        }
    }
})();
```

### 2. Cliente en React para forma de archivo

```javascript
import React, { useState } from 'react';
import axios from 'axios';
import { v4 as uuidv4 } from 'uuid';

const BiometriaForm = () => {
    const [files, setFiles] = useState({
        cedula: null,
        rostro: null,
        registro: null
    });
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);

    const encodeFileToBase64 = (file) => {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    };

    const handleFileChange = (e, fileType) => {
        setFiles(prev => ({
            ...prev,
            [fileType]: e.target.files[0]
        }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        
        if (!files.cedula || !files.rostro) {
            alert('Por favor selecciona cédula y rostro');
            return;
        }

        setLoading(true);

        try {
            const cedulaB64 = await encodeFileToBase64(files.cedula);
            const rostroB64 = await encodeFileToBase64(files.rostro);

            const payload = {
                uuidProceso: uuidv4(),
                cedulaFrontalBase64: cedulaB64,
                rostroPersonaBase64: rostroB64
            };

            if (files.registro) {
                payload.registroCivilBase64 = await encodeFileToBase64(files.registro);
            }

            const response = await axios.post(
                'http://localhost:8000/api/biometrics/demo_validation_extended',
                payload
            );

            setResult(response.data);
        } catch (error) {
            setResult({
                status: false,
                message: `Error: ${error.message}`
            });
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="biometria-form">
            <h1>Validación Biométrica Extendida</h1>
            
            <form onSubmit={handleSubmit}>
                <div className="form-group">
                    <label>Cédula Frontal *</label>
                    <input
                        type="file"
                        accept="image/*"
                        onChange={(e) => handleFileChange(e, 'cedula')}
                        required
                    />
                </div>

                <div className="form-group">
                    <label>Rostro Persona *</label>
                    <input
                        type="file"
                        accept="image/*"
                        onChange={(e) => handleFileChange(e, 'rostro')}
                        required
                    />
                </div>

                <div className="form-group">
                    <label>Registro Civil (Opcional)</label>
                    <input
                        type="file"
                        accept="image/*"
                        onChange={(e) => handleFileChange(e, 'registro')}
                    />
                </div>

                <button type="submit" disabled={loading}>
                    {loading ? 'Procesando...' : 'Validar'}
                </button>
            </form>

            {result && (
                <div className={`result ${result.status ? 'success' : 'error'}`}>
                    <h2>{result.status ? '✅ Validación Exitosa' : '❌ Validación Fallida'}</h2>
                    <p>{result.message}</p>
                    
                    {result.data && (
                        <details>
                            <summary>Detalles de los Scores</summary>
                            <pre>{JSON.stringify(result.data, null, 2)}</pre>
                        </details>
                    )}
                </div>
            )}
        </div>
    );
};

export default BiometriaForm;
```

---

## Prueba Rápida (cURL simple)

```bash
# 1. Codificar una imagen de prueba
CEDULA_B64=$(base64 -w 0 < cedula.jpg | sed 's/^/data:image\/jpeg;base64,/')
ROSTRO_B64=$(base64 -w 0 < rostro.jpg | sed 's/^/data:image\/jpeg;base64,/')

# 2. Hacer la solicitud
curl -X POST http://localhost:8000/api/biometrics/demo_validation_extended \
  -H "Content-Type: application/json" \
  -d "{
    \"uuidProceso\": \"$(uuidgen)\",
    \"cedulaFrontalBase64\": \"${CEDULA_B64}\",
    \"rostroPersonaBase64\": \"${ROSTRO_B64}\"
  }" | python -m json.tool
```

---

## Notas Importantes

1. **Tamaño de solicitud**: Las imágenes en base64 son voluminosas. Asegúrate de que tu servidor permita payloads grandes.

2. **Timeout**: El procesamiento puede tomar 5-10 segundos. Configura un timeout adecuado.

3. **Formato de imagen**: Usa JPEGs con buena claridad. Imágenes de mala calidad resultan en fallos de detección.

4. **Gestión de errores**: Siempre verifica `response.status` antes de confiar en los datos.

5. **UUID del proceso**: Usa el mismo `uuidProceso` para múltiples validaciones del mismo usuario.

---

## Troubleshooting

**Problema**: "registroCivilBase64 inválido"
- **Solución**: Asegúrate de incluir el prefijo `data:image/jpeg;base64,` antes del contenido.

**Problema**: "No se detectó rostro"
- **Solución**: Usa imágenes más claras, mejor iluminadas y con rostros frontales.

**Problema**: Timeout
- **Solución**: Aumenta el timeout del cliente. El servicio puede tardar 10-15 segundos.

**Problema**: Error 500
- **Solución**: Revisa los logs del servidor en `/biometria_flows/`.

