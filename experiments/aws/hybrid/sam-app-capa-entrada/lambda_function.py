import base64
import json
import os

import requests


CONTAINER_URL = os.environ.get("CONTAINER_URL")


def _response(status_code: int, body):
    if isinstance(body, dict):
        body = json.dumps(body, ensure_ascii=False, indent=2)

    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": body,
    }


def lambda_handler(event, context):
    method = (
        event.get("httpMethod")
        or event.get("requestContext", {}).get("http", {}).get("method")
        or "GET"
    ).upper()

    if method == "GET":
        return _response(
            200,
            "Funcion capa_entrada activa (AWS llama a contenedor)",
        )

    if method == "POST":
        if not CONTAINER_URL:
            return _response(
                500,
                {"error": "Falta la variable de entorno CONTAINER_URL."},
            )

        try:
            body_raw = event.get("body") or "{}"

            if event.get("isBase64Encoded", False):
                body_raw = base64.b64decode(body_raw).decode("utf-8")

            response = requests.post(
                CONTAINER_URL,
                headers={"Content-Type": "application/json"},
                data=body_raw,
            )

            return _response(response.status_code, response.json())

        except Exception as error:
            return _response(
                500,
                f"Error llamando al contenedor: {error}",
            )

    return _response(405, "Método no soportado")