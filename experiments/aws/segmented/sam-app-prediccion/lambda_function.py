import os, json, time, base64, logging, tempfile, csv
import boto3
import requests
import numpy as np
import psutil

logging.basicConfig(level=logging.INFO)

# ENV
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "red-neuronal-segmentada-modelos")
TRAINED_KEY    = os.environ.get("TRAINED_KEY",    "modelo_entrenado.h5")
CONTAINER_PREDICT_URL = os.environ.get("CONTAINER_PREDICT_URL", "")  # .../predict-model

s3 = boto3.client("s3")

def _resp(code, body):
    if isinstance(body, (dict, list)):
        body = json.dumps(body, ensure_ascii=False, indent=2)
    return {"statusCode": code, "headers": {"Content-Type":"application/json"}, "body": body}

def _method(event):
    return (event.get("httpMethod")
            or event.get("requestContext", {}).get("http", {}).get("method")
            or "GET").upper()

def lambda_handler(event, context):
    if _method(event) == "GET":
        return _resp(200, "Función predicción activa (AWS orquesta; contenedor predice)")

    # ---- Parse body ----
    try:
        body_raw = event.get("body") or "{}"
        if event.get("isBase64Encoded"):
            body_raw = base64.b64decode(body_raw).decode("utf-8")
        body = json.loads(body_raw)
    except Exception:
        return _resp(400, {"error": "Body JSON inválido"})

    if "entrada" not in body:
        return _resp(400, {"error": "Falta 'entrada'. Ej: {\"entrada\":[1.0,2.0]} o [[1.0,2.0],[2.0,3.0]]"})

    # Normalizar entrada a 2D
    entrada = np.array(body["entrada"], dtype=float)
    if entrada.ndim == 1:
        entrada = entrada.reshape(1, -1)

    # ---- Métricas inicio ----
    tiempos = {}
    t_total = time.perf_counter()
    proc = psutil.Process(os.getpid())
    cpu_ini = proc.cpu_percent(interval=None)
    mem_ini = proc.memory_info().rss/(1024**2)
    disk_ini = 0.0  # no persistimos más allá de /tmp

    # ---- Descargar modelo entrenado desde S3 ----
    tmp = tempfile.gettempdir()
    local_trained = os.path.join(tmp, "modelo_entrenado.h5")

    t = time.perf_counter()
    try:
        s3.download_file(S3_BUCKET_NAME, TRAINED_KEY, local_trained)
    except Exception as e:
        return _resp(500, {"error": f"No se pudo descargar {TRAINED_KEY} de S3: {str(e)}"})
    tiempos["descargar_modelo_s"] = round(time.perf_counter() - t, 3)

    # ---- Leer modelo y llamar al contenedor /predict-model ----
    if not CONTAINER_PREDICT_URL:
        return _resp(500, {"error": "Falta variable de entorno CONTAINER_PREDICT_URL (…/predict-model)"})

    with open(local_trained, "rb") as f:
        model_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model_base64": model_b64,
        "X": entrada.tolist()
    }

    t = time.perf_counter()
    r = requests.post(CONTAINER_PREDICT_URL, json=payload, timeout=60)
    tiempos["predecir_s"] = round(time.perf_counter() - t, 3)
    if r.status_code != 200:
        return _resp(500, {"error": r.text, "tiempos": tiempos})

    out = r.json()
    preds = out.get("predictions")
    if preds is None:
        return _resp(500, {"error": "El contenedor no devolvió 'predictions'", "tiempos": tiempos})

    # ---- Métricas fin ----
    cpu_fin = proc.cpu_percent(interval=None)
    mem_fin = proc.memory_info().rss/(1024**2)
    tiempos["total_s"] = round(time.perf_counter() - t_total, 3)

    # Si era 1 sola muestra, devolver escalar como en Azure
    if len(entrada) == 1 and isinstance(preds, list) and len(preds) > 0:
        try:
            valor_predicho = float(preds[0][0])
        except Exception:
            valor_predicho = float(preds[0]) if isinstance(preds[0], (int,float)) else preds[0]
        body_out = {
            "prediccion": valor_predicho,
            "tiempos": {
                **tiempos,
                "memoria_inicial_mb": round(mem_ini, 2),
                "memoria_final_mb": round(mem_fin, 2),
                "cpu_inicial_percent": cpu_ini,
                "cpu_final_percent": cpu_fin
            }
        }
    else:
        body_out = {
            "predicciones": preds,
            "tiempos": {
                **tiempos,
                "memoria_inicial_mb": round(mem_ini, 2),
                "memoria_final_mb": round(mem_fin, 2),
                "cpu_inicial_percent": cpu_ini,
                "cpu_final_percent": cpu_fin
            }
        }

    return _resp(200, body_out)
