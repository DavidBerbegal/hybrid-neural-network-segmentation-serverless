import os
import json
import time
import base64
import logging
import tempfile

import boto3
import requests
import numpy as np
import pandas as pd
import psutil


logging.basicConfig(level=logging.INFO)

S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "")
UNTRAINED_KEY = os.environ.get(
    "UNTRAINED_KEY", "modelo_sin_entrenar.h5"
)
TRAINED_KEY = os.environ.get(
    "TRAINED_KEY", "modelo_entrenado.h5"
)
DATOS_KEY = os.environ.get(
    "DATOS_KEY", "datos_entrenamiento.csv"
)
CONTAINER_TRAIN_URL = os.environ.get("CONTAINER_TRAIN_URL", "")

temp_dir = tempfile.gettempdir()
LOCAL_UNTRAINED_MODEL_PATH = os.path.join(
    temp_dir, "modelo_sin_entrenar.h5"
)
LOCAL_TRAINED_MODEL_PATH = os.path.join(
    temp_dir, "modelo_entrenado.h5"
)
LOCAL_CSV_PATH = os.path.join(
    temp_dir, "datos_entrenamiento.csv"
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


def uso_disco_temporal():
    total = 0

    for root, _, files in os.walk(tempfile.gettempdir()):
        for filename in files:
            try:
                total += os.path.getsize(
                    os.path.join(root, filename)
                )
            except Exception:
                pass

    return total / (1024**2)


def _get_body_params(event, method):
    if method == "GET":
        return dict(event.get("queryStringParameters") or {})

    try:
        body_raw = event.get("body") or "{}"

        if event.get("isBase64Encoded", False):
            body_raw = base64.b64decode(body_raw).decode("utf-8")

        return json.loads(body_raw)
    except Exception:
        return {}


def lambda_handler(event, context):
    method = _method(event)

    if method not in ("GET", "POST"):
        return _resp(405, "Método no soportado")

    tiempos = {}
    start_total = time.perf_counter()
    proc = psutil.Process(os.getpid())
    cpu_ini = proc.cpu_percent(interval=None)
    mem_ini = proc.memory_info().rss / (1024**2)
    disk_ini = uso_disco_temporal()

    for key, value in {
        "S3_BUCKET_NAME": S3_BUCKET_NAME,
        "CONTAINER_TRAIN_URL": CONTAINER_TRAIN_URL,
    }.items():
        if not value:
            return _resp(
                500,
                {"error": f"Falta variable de entorno {key}."},
            )

    body = _get_body_params(event, method)

    try:
        total_epochs = int(body.get("total_epochs"))
        epoch_inicio = int(body.get("epoch_inicio", 0))
        block_size = int(body.get("block_size"))
        batch_size = int(body.get("batch_size"))
    except Exception as error:
        return _resp(
            400,
            {
                "error": (
                    "Parámetros inválidos. Obligatorios: total_epochs, "
                    "block_size, batch_size. Opcional: epoch_inicio."
                ),
                "detalle": str(error),
                "body": body,
            },
        )

    epoch_final = min(epoch_inicio + block_size, total_epochs)

    for path in (
        LOCAL_UNTRAINED_MODEL_PATH,
        LOCAL_TRAINED_MODEL_PATH,
        LOCAL_CSV_PATH,
    ):
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    try:
        t = time.perf_counter()

        try:
            s3.download_file(
                S3_BUCKET_NAME,
                TRAINED_KEY,
                LOCAL_UNTRAINED_MODEL_PATH,
            )
            modelo_origen = TRAINED_KEY
        except Exception:
            s3.download_file(
                S3_BUCKET_NAME,
                UNTRAINED_KEY,
                LOCAL_UNTRAINED_MODEL_PATH,
            )
            modelo_origen = UNTRAINED_KEY

        tiempos["descargar_modelo_s"] = round(
            time.perf_counter() - t, 3
        )

        t = time.perf_counter()

        s3.download_file(
            S3_BUCKET_NAME,
            DATOS_KEY,
            LOCAL_CSV_PATH,
        )

        tiempos["descargar_csv_s"] = round(
            time.perf_counter() - t, 3
        )

        t = time.perf_counter()

        df = pd.read_csv(LOCAL_CSV_PATH, nrows=batch_size)
        X = df.iloc[:, :2].astype(np.float32).values
        Y = df.iloc[:, 2].astype(np.float32).values.reshape(-1, 1)

        tiempos["cargar_csv_s"] = round(
            time.perf_counter() - t, 3
        )

        t = time.perf_counter()

        with open(LOCAL_UNTRAINED_MODEL_PATH, "rb") as file:
            model_b64 = base64.b64encode(
                file.read()
            ).decode("utf-8")

        payload = {
            "model_base64": model_b64,
            "X": X.tolist(),
            "Y": Y.tolist(),
            "epochs": epoch_final,
            "initial_epoch": epoch_inicio,
        }

        tiempos["preparar_payload_s"] = round(
            time.perf_counter() - t, 3
        )

        t = time.perf_counter()

        response = requests.post(
            CONTAINER_TRAIN_URL,
            json=payload,
            timeout=900,
        )

        tiempos["cont_train_s"] = round(
            time.perf_counter() - t, 3
        )

        if response.status_code != 200:
            return _resp(
                500,
                {"error": response.text, "tiempos": tiempos},
            )

        out = response.json()
        trained_b64 = out.get("model_base64")

        if not trained_b64:
            return _resp(
                500,
                {
                    "error": "El contenedor no devolvió 'model_base64'",
                    "tiempos": tiempos,
                },
            )

        tiempos_contenedor = out["tiempos"]

        # En Azure este campo incluye carga y recompilación.
        tiempos["compilar_modelo_s"] = round(
            tiempos_contenedor["cargar_modelo_s"]
            + tiempos_contenedor["compilar_modelo_s"],
            3,
        )

        # Tiempo interno de fit, sin comunicación ni serialización.
        tiempos["entrenamiento_s"] = out["train_time_s"]

        t = time.perf_counter()
        trained_bytes = base64.b64decode(trained_b64)

        with open(LOCAL_TRAINED_MODEL_PATH, "wb") as file:
            file.write(trained_bytes)

        tiempos["guardar_h5_recibido_s"] = round(
            time.perf_counter() - t, 3
        )

        t = time.perf_counter()

        s3.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=TRAINED_KEY,
            Body=trained_bytes,
        )

        tiempos["subir_modelo_s"] = round(
            time.perf_counter() - t, 3
        )

        cpu_fin = proc.cpu_percent(interval=None)
        mem_fin = proc.memory_info().rss / (1024**2)
        disk_fin = uso_disco_temporal()

        tiempos.update({
            "modelo_origen": modelo_origen,
            "cpu_inicial_percent": cpu_ini,
            "cpu_final_percent": cpu_fin,
            "memoria_inicial_mb": round(mem_ini, 2),
            "memoria_final_mb": round(mem_fin, 2),
            "memoria_consumida_mb": round(mem_fin - mem_ini, 2),
            "disk_inicial_mb": round(disk_ini, 2),
            "disk_final_mb": round(disk_fin, 2),
            "disk_consumido_mb": round(disk_fin - disk_ini, 2),
            "total_s": round(time.perf_counter() - start_total, 3),
        })

        return _resp(
            200,
            {
                "bloque_completado": [epoch_inicio, epoch_final],
                "epocas_totales": total_epochs,
                "tiempos": tiempos,
                "tiempos_contenedor": tiempos_contenedor,
                "recursos_contenedor": out.get("recursos_contenedor"),
                "metricas_finales": out.get("metrics", {}),
                "registros_totales": int(X.shape[0]),
            },
        )

    except Exception as error:
        logging.exception("Error durante el entrenamiento")
        return _resp(
            500,
            {"error": str(error), "tiempos": tiempos},
        )