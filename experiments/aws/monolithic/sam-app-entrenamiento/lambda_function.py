import json
import os
import time
import tempfile
import base64
import logging
import psutil
import numpy as np
import requests
import boto3

# ========= Config =========
# URL del contenedor (Function URL o endpoint del API Gateway que expone Flask/awsgi)
CONTAINER_BASE_URL = os.getenv("CONTAINER_BASE_URL", "<TENSORFLOW_CONTAINER_URL>")

# S3
S3_BUCKET = os.getenv("S3_BUCKET", "red-neuronal-modelos")
TRAINED_MODEL_KEY = os.getenv("TRAINED_MODEL_KEY", "modelo_entrenado.h5")

# ======== Logging =========
logger = logging.getLogger()
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")

# ========= Utilidades =========
def obtener_datos(n=1000):
    np.random.seed(42)
    X = np.random.uniform(0, 10, (n, 2))
    Y = (X[:, 0] + X[:, 1]).reshape(-1, 1)
    return X, Y

def uso_disco_temporal():
    temp_dir = tempfile.gettempdir()
    total_size = 0
    for dirpath, _, filenames in os.walk(temp_dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total_size += os.path.getsize(fp)
            except Exception:
                pass
    return total_size / (1024 ** 2)  # MB

def http_post_json(url, payload):
    r = requests.post(url, json=payload, timeout=900)
    r.raise_for_status()
    return r.json()

def guardar_en_s3(local_path, bucket, key):
    with open(local_path, "rb") as f:
        s3.upload_fileobj(f, bucket, key)

# ========= Construcción de modelo vía contenedor =========
def construir_modelo_via_contenedor(num_capas_ocultas: int):
    # 1) capa de entrada
    entrada = http_post_json(
        f"{CONTAINER_BASE_URL}/capa-entrada",
        {
            "units": 5,
            "activation": "relu",
            "input_shape": [2],
            "name": "entrada",
        },
    )

    # 2) capas ocultas
    ocultas = []
    for i in range(num_capas_ocultas):
        ocultas.append(
            http_post_json(
                f"{CONTAINER_BASE_URL}/capa-oculta",
                {
                    "units": 4,
                    "activation": "relu",
                    "name": f"oculta_{i+1}",
                },
            )
        )

    # 3) capa de salida
    salida = http_post_json(
        f"{CONTAINER_BASE_URL}/capa-salida",
        {
            "units": 1,
            "activation": "linear",
            "name": "salida",
        },
    )

    # 4) ensamblar en /configure-layer
    layers = [entrada] + ocultas + [salida]
    cfg_resp = http_post_json(
        f"{CONTAINER_BASE_URL}/configure-layer",
        {"layers": layers},
    )
    # Devuelve {"model_base64": "..."}
    return cfg_resp["model_base64"]

# ========= Entrenamiento vía contenedor (AHORA devuelve modelo + métricas) =========
def entrenar_via_contenedor(model_b64: str, X, Y, total_epochs, epoch_inicio):
    payload = {
        "model_base64": model_b64,
        "X": np.asarray(X, dtype=float).tolist(),
        "Y": np.asarray(Y, dtype=float).tolist(),
        "epochs": int(total_epochs),
        "initial_epoch": int(epoch_inicio),
        # Paridad con Azure
        "optimizer": "sgd",
        "loss": "mean_squared_error",
        "metrics": ["mae", "mse", "mape"],
    }
    resp = http_post_json(f"{CONTAINER_BASE_URL}/train-model", payload)
    # El contenedor devuelve: model_base64, metrics (dict), train_time_s (float)
    return resp["model_base64"], resp.get("metrics", {}), resp.get("train_time_s")

# ========= Handler principal (API Gateway/Lambda URL proxy) =========
def lambda_handler(event, context):
    # Soporta API Gateway HTTP/API o Function URL
    try:
        body = event.get("body")
        if body and event.get("isBase64Encoded"):
            body = base64.b64decode(body).decode("utf-8")
        data = json.loads(body or "{}")
    except Exception as e:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": f"Error en parámetros de entrada: {str(e)}"}),
        }

    tiempos = {}
    entrenamiento_batch = []
    metricas = {}
    t_total_inicio = time.perf_counter()
    process = psutil.Process(os.getpid())

    # MONITORIZACIÓN INICIAL
    memoria_inicial = process.memory_info().rss / 1024**2
    disco_inicial = uso_disco_temporal()
    cpu_inicial = process.cpu_percent(interval=None)

    # PARÁMETROS (igual que en Azure)
    try:
        total_epochs      = int(data["total_epochs"])
        epoch_inicio      = int(data.get("epoch_inicio", 0))
        block_size        = int(data.get("block_size", 100))
        num_capas_ocultas = int(data.get("num_capas_ocultas", 1))
        n_datos           = int(data.get("n_datos", 1000))
    except Exception as e:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "Error en parámetros de entrada: " + str(e)}),
        }

    epoch_final = min(epoch_inicio + block_size, total_epochs)

    # GENERACIÓN DE DATOS
    t1 = time.perf_counter()
    X, Y = obtener_datos(n_datos)
    tiempos["generacion_datos_s"] = round(time.perf_counter() - t1, 3)

    # CONSTRUCCIÓN DE MODELO EN EL CONTENEDOR
    t1 = time.perf_counter()
    try:
        model_b64 = construir_modelo_via_contenedor(num_capas_ocultas)
    except Exception as e:
        logger.exception("Fallo construyendo modelo en contenedor")
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": f"Error construyendo modelo en contenedor: {str(e)}"}),
        }
    tiempos["compilacion_modelo_s"] = round(time.perf_counter() - t1, 3)

    # ENTRENAMIENTO EN EL CONTENEDOR (ahora captura métricas)
    t1 = time.perf_counter()
    try:
        trained_b64, metricas, train_time_s = entrenar_via_contenedor(
            model_b64, X, Y, epoch_final, epoch_inicio
        )
    except Exception as e:
        logger.exception("Fallo entrenando en contenedor")
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": f"Error entrenando modelo en contenedor: {str(e)}"}),
        }
    tiempos["entrenamiento_s"] = round(time.perf_counter() - t1, 3)
    # Si prefieres el tiempo interno del contenedor, descomenta:
    # if train_time_s is not None:
    #     tiempos["entrenamiento_s"] = float(train_time_s)

    # GUARDADO Y SUBIDA A S3
    t1 = time.perf_counter()
    try:
        raw = base64.b64decode(trained_b64)
        temp_path = os.path.join(tempfile.gettempdir(), "modelo_entrenado.h5")
        with open(temp_path, "wb") as f:
            f.write(raw)
        guardar_en_s3(temp_path, S3_BUCKET, TRAINED_MODEL_KEY)
    except Exception as e:
        logger.exception("Fallo guardando/subiendo a S3")
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": f"Error guardando/subiendo a S3: {str(e)}"}),
        }
    tiempos["guardado_subida_modelo_s"] = round(time.perf_counter() - t1, 3)

    # MONITORIZACIÓN FINAL
    memoria_final = process.memory_info().rss / 1024**2
    disco_final   = uso_disco_temporal()
    cpu_final     = process.cpu_percent(interval=None)
    tiempos.update({
        "memoria_inicial_mb": round(memoria_inicial, 2),
        "memoria_final_mb":   round(memoria_final, 2),
        "memoria_consumida_mb": round(memoria_final - memoria_inicial, 2),
        "disco_inicial_mb":   round(disco_inicial, 2),
        "disco_final_mb":     round(disco_final, 2),
        "disco_consumido_mb": round(disco_final - disco_inicial, 2),
        "cpu_inicial_percent": cpu_inicial,
        "cpu_final_percent":   cpu_final,
        "total_s": round(time.perf_counter() - t_total_inicio, 3)
    })

    # Respuesta con la misma forma que en Azure
    resp = {
        "bloque_completado": [epoch_inicio, epoch_final],
        "epocas_totales": total_epochs,
        "tiempos": tiempos,
        "metricas_finales": metricas,
        "entrenamiento_batch": [
            {
                "batch": 0,
                "registros": int(X.shape[0]),
                "metricas": metricas,
                "memoria_pre_batch_mb": round(memoria_inicial, 2),
                "memoria_post_batch_mb": round(memoria_final, 2),
                "cpu_pre_batch_percent": cpu_inicial,
                "cpu_post_batch_percent": cpu_final,
                "tiempo_batch_s": tiempos["entrenamiento_s"],
            }
        ],
        "registros_totales": int(X.shape[0]),
    }

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(resp, ensure_ascii=False, indent=2),
    }
