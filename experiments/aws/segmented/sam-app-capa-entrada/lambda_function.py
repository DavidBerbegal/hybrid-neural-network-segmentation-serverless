import json
import os
import requests

CONTAINER_URL = os.environ.get("CONTAINER_URL", "<TENSORFLOW_CONTAINER_URL>/capa-entrada")

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

    # GET: simple ping
    if method == "GET":
        return _response(200, "Funcion capa_entrada activa (AWS llama a contenedor)")

    # POST: reenviar al contenedor
    if method == "POST":
        try:
            body_raw = event.get("body") or "{}"
            headers = {"Content-Type": "application/json"}
            r = requests.post(CONTAINER_URL, headers=headers, data=body_raw)
            return _response(r.status_code, r.json())
        except Exception as e:
            return _response(500, f"Error llamando al contenedor: {str(e)}")

    return _response(405, "Método no soportado")
