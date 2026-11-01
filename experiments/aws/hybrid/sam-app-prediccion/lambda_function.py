import os
import json
import time
import base64
import logging
import tempfile

import boto3
import requests
import numpy as np
import psutil


logging.basicConfig(level=logging.INFO)

S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "")
TRAINED_KEY = os.environ.get(
    "TRAINED_KEY", "modelo_entrenado.h5"
)
CONTAINER_PREDICT_URL = os.environ.get(
    "CONTAINER_PREDICT_URL", ""
)

LOCAL_TRAINED_MODEL_PATH = os.path.join(
    tempfile.gettempdir(), "modelo_entrenado.h5"
)

s3 = boto3.client("s3")


def _resp(code, body):
    if isinstance(body, (dict, list)):
        body = json.dumps(body, ensure_ascii=False, indent=2)

    return {
        "statusCode": code,
        "headers": {"Content-Type": "application/json"},
        "body": body,
    }


def _method(event):
    return (
        event.get("httpMethod")
        or event.get("requestContext", {}).get("http", {}).get("method")
        or "GET"
    ).upper()


def lambda_handler(event, context):
    if _method(event) != "POST":
        return _resp(405, "Método no soportado")

    tiempos = {}
    start_total = time.perf_counter()
    proc = psutil.Process(os.getpid())
    mem_ini = proc.memory_info().rss / (1024**2)
    cpu_ini = proc.cpu_percent(interval=None)

    for key, value in {
        "S3_BUCKET_NAME": S3_BUCKET_NAME,
        "CONTAINER_PREDICT_URL": CONTAINER_PREDICT_URL,
    }.items():
        if not value:
            return _resp(
                500,
                {"error": f"Falta variable de entorno {key}."},
            )

    try:
        t = time.perf_counter()

        try:
            try:
                if os.path.exists(LOCAL_TRAINED_MODEL_PATH):
                    os.remove(LOCAL_TRAINED_MODEL_PATH)
            except Exception:
                pass

            s3.download_file(
                S3_BUCKET_NAME,
                TRAINED_KEY,
                LOCAL_TRAINED_MODEL_PATH,
            )

            tiempos["descargar_modelo_s"] = round(
                time.perf_counter() - t, 3
            )

        except Exception as error:
            tiempos["descargar_modelo_s"] = round(
                time.perf_counter() - t, 3
            )

            return _resp(
                500,
                {
                    "error": f"Error al descargar el modelo: {error}",
                    "tiempos": tiempos,
                },
            )

        t = time.perf_counter()

        try:
            body_raw = event.get("body") or ""

            if event.get("isBase64Encoded", False):
                body_raw = base64.b64decode(body_raw).decode("utf-8")

            body = json.loads(body_raw)

            if "entrada" not in body:
                raise ValueError(
                    "Falta el campo 'entrada' en el body."
                )

            entrada = np.array(body["entrada"], dtype=np.float32)

            if entrada.ndim == 1:
                entrada = entrada.reshape(1, -1)

            if entrada.ndim != 2:
                raise ValueError(
                    "El campo 'entrada' debe ser un vector 1D "
                    "o una matriz 2D."
                )

            tiempos["procesar_entrada_s"] = round(
                time.perf_counter() - t, 3
            )

        except Exception as error:
            tiempos["procesar_entrada_s"] = round(
                time.perf_counter() - t, 3
            )

            return _resp(
                400,
                {
                    "error": f"Error al procesar la entrada: {error}",
                    "tiempos": tiempos,
                },
            )

        t = time.perf_counter()

        with open(LOCAL_TRAINED_MODEL_PATH, "rb") as file:
            model_b64 = base64.b64encode(
                file.read()
            ).decode("utf-8")

        payload = {
            "model_base64": model_b64,
            "X": entrada.tolist(),
        }

        tiempos["preparar_payload_s"] = round(
            time.perf_counter() - t, 3
        )

        t = time.perf_counter()

        response = requests.post(
            CONTAINER_PREDICT_URL,
            json=payload,
            timeout=60,
        )

        tiempos["cont_predict_s"] = round(
            time.perf_counter() - t, 3
        )

        if response.status_code != 200:
            return _resp(
                500,
                {"error": response.text, "tiempos": tiempos},
            )

        out = response.json()
        predictions = out.get("predictions")

        if predictions is None:
            return _resp(
                500,
                {
                    "error": "El contenedor no devolvió 'predictions'",
                    "tiempos": tiempos,
                },
            )

        tiempos_contenedor = out["tiempos"]
        tiempos["cargar_modelo_s"] = (
            tiempos_contenedor["cargar_modelo_s"]
        )
        tiempos["predecir_s"] = out["predict_time_s"]

        # Azure devuelve únicamente el primer valor.
        valor = float(np.array(predictions).reshape(-1)[0])

        mem_fin = proc.memory_info().rss / (1024**2)
        cpu_fin = proc.cpu_percent(interval=None)

        tiempos.update({
            "memoria_inicial_mb": round(mem_ini, 2),
            "memoria_final_mb": round(mem_fin, 2),
            "memoria_consumida_mb": round(mem_fin - mem_ini, 2),
            "cpu_inicial_percent": cpu_ini,
            "cpu_final_percent": cpu_fin,
            "total_s": round(time.perf_counter() - start_total, 3),
        })

        return _resp(
            200,
            {
                "prediccion": valor,
                "tiempos": tiempos,
                "tiempos_contenedor": tiempos_contenedor,
                "recursos_contenedor": out.get("recursos_contenedor"),
            },
        )

    except Exception as error:
        logging.exception("Error durante la predicción")

        return _resp(
            500,
            {"error": str(error), "tiempos": tiempos},
        )