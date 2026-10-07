import json
import os
import requests

CONTAINER_URL = os.environ.get(
    "CONTAINER_URL",
    "<TENSORFLOW_CONTAINER_URL>/capa-oculta"
)

def _response(status_code: int, body):
    if isinstance(body, dict):
        body = json.dumps(body, ensure_ascii=False, indent=2)
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": body
    }

def lambda_handler(event, context):
    method = (event.get("httpMethod")
              or event.get("requestContext", {}).get("http", {}).get("method")
              or "GET").upper()

    if method == "GET":
        return _response(200, "Función capa_oculta activa (AWS llama a contenedor)")

    if method == "POST":
        try:
            body_raw = event.get("body") or "{}"
            r = requests.post(CONTAINER_URL, headers={"Content-Type": "application/json"}, data=body_raw)
            # r.json() puede lanzar; capturamos texto si no es JSON
            try:
                payload = r.json()
            except Exception:
                payload = r.text
            return _response(r.status_code, payload)
        except Exception as e:
            return _response(500, f"Error llamando al contenedor: {str(e)}")

    return _response(405, "Método no soportado")
