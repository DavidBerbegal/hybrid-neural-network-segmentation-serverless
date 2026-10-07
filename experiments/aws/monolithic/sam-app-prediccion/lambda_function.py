import json
import os
import time
import base64
import logging
import tempfile
import psutil
import boto3
import requests

# === Config ===
CONTAINER_BASE_URL = os.getenv(
    "CONTAINER_BASE_URL",
    "<TENSORFLOW_CONTAINER_URL>"  # sin barra final
)

S3_BUCKET = os.getenv("S3_BUCKET", "red-neuronal-monolitica-modelos")
TRAINED_MODEL_KEY = os.getenv("TRAINED_MODEL_KEY", "modelo_entrenado.h5")

logger = logging.getLogger()
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")

def http_post_json(url, payload, timeout=60):
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

def descargar_modelo_b64_desde_s3(bucket: str, key: str) -> str:
    t0 = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
        s3.download_fileobj(bucket, key, tmp)
        tmp.flush()
        tmp.seek(0)
        raw = tmp.read()
    t = round(time.perf_counter() - t0, 3)
    return base64.b64encode(raw).decode("utf-8"), t

def lambda_handler(event, context):
    tiempos = {}
    t_total_inicio = time.perf_counter()
    process = psutil.Process(os.getpid())

    # MONITORIZACIÓN INICIAL
    memoria_inicial = process.memory_info().rss / 1024**2
    cpu_inicial = process.cpu_percent(interval=None)

    # Entrada (igual que Azure: { "entrada": [a, b] })
    try:
        body = event.get("body")
        if body and event.get("isBase64Encoded"):
            body = base64.b64decode(body).decode("utf-8")
        data = json.loads(body or "{}")
        entrada = data["entrada"]
        # Normalización ligera: debe ser lista (1D) de longitud 2
        if not isinstance(entrada, list):
            raise ValueError("El campo 'entrada' debe ser una lista.")
    except Exception as e:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": f"Parámetros inválidos. Esperado: {{'entrada':[a,b]}}. Detalle: {e}"})
        }

    if not CONTAINER_BASE_URL:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "CONTAINER_BASE_URL no configurado"})
        }

    # Paso 1: Descargar modelo (S3)  -> tiempos.descargar_modelo
    try:
        model_b64, t_desc = descargar_modelo_b64_desde_s3(S3_BUCKET, TRAINED_MODEL_KEY)
        tiempos["descargar_modelo"] = t_desc
    except Exception as e:
        logger.exception("Error descargando el modelo de S3")
        tiempos["descargar_modelo"] = 0.0
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": f"Error descargando modelo de S3: {e}", "tiempos": tiempos})
        }

    # Paso 2: Cargar modelo -> en Azure se hace local; aquí lo carga el contenedor.
    # Para mantener el formato, dejamos el campo con 0.0
    tiempos["cargar_modelo"] = 0.0

    # Paso 3: Procesar entrada -> tiempos.procesar_entrada
    t1 = time.perf_counter()
    # El contenedor acepta X como lista; si llega 1D, él ya convierte a 2D
    X = entrada
    tiempos["procesar_entrada"] = round(time.perf_counter() - t1, 3)

    # Paso 4: Predicción en el contenedor -> tiempos.predecir
    try:
        t2 = time.perf_counter()
        resp = http_post_json(
            f"{CONTAINER_BASE_URL}/predict-model",
            {"model_base64": model_b64, "X": X},
            timeout=60
        )
        tiempos["predecir"] = round(time.perf_counter() - t2, 3)
    except Exception as e:
        logger.exception("Error en predicción del contenedor")
        tiempos["predecir"] = round(time.perf_counter() - t2, 3)
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": f"Error en predicción del contenedor: {e}", "tiempos": tiempos})
        }

    # Extraer predicción (igualar a Azure: valor escalar)
    preds = resp.get("predictions", [])
    if preds and isinstance(preds[0], list):
        valor_predicho = float(preds[0][0])
    elif preds:
        valor_predicho = float(preds[0])
    else:
        valor_predicho = None

    # MONITORIZACIÓN FINAL (mismas claves que Azure)
    memoria_final = process.memory_info().rss / 1024**2
    cpu_final = process.cpu_percent(interval=None)
    tiempos["memoria_inicial_mb"] = round(memoria_inicial, 2)
    tiempos["memoria_final_mb"] = round(memoria_final, 2)
    tiempos["cpu_inicial_percent"] = cpu_inicial
    tiempos["cpu_final_percent"] = cpu_final
    tiempos["total"] = round(time.perf_counter() - t_total_inicio, 3)

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({
            "prediccion": valor_predicho,
            "tiempos": tiempos
        }, ensure_ascii=False, indent=2)
    }
