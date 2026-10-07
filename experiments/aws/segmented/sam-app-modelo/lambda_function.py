import os
import json
import time
import base64
import logging
import requests
import boto3
import psutil

logging.basicConfig(level=logging.INFO)

# URLs (Function URLs) — configúralas vía variables de entorno
URL_CAPA_ENTRADA = os.environ.get("URL_CAPA_ENTRADA", "")
URL_CAPA_OCULTA  = os.environ.get("URL_CAPA_OCULTA", "")
URL_CAPA_SALIDA  = os.environ.get("URL_CAPA_SALIDA", "")
URL_CONTENEDOR_CONFIG = os.environ.get("URL_CONTENEDOR_CONFIG", "")  # .../configure-layer
S3_BUCKET_NAME   = os.environ.get("S3_BUCKET_NAME", "red-neuronal-segmentada-modelos")
S3_KEY_H5        = os.environ.get("S3_KEY_H5", "modelo_sin_entrenar.h5")

s3 = boto3.client("s3")

def _resp(status, body):
    if isinstance(body, (dict, list)):
        body = json.dumps(body, ensure_ascii=False, indent=2)
    return {"statusCode": status, "headers": {"Content-Type": "application/json"}, "body": body}

def _parse_method(event):
    return (event.get("httpMethod")
            or event.get("requestContext", {}).get("http", {}).get("method")
            or "GET").upper()

def lambda_handler(event, context):
    method = _parse_method(event)
    if method == "GET":
        return _resp(200, "Función modelo activa (AWS orquesta y el contenedor crea el .h5)")

    if method != "POST":
        return _resp(405, "Método no soportado")

    # ---- Monitoreo inicial (como en Azure) ----
    t0 = time.perf_counter()
    proc = psutil.Process()
    cpu_ini = proc.cpu_percent(interval=None)
    mem_ini = proc.memory_info().rss / (1024**2)  # MB
    tiempos = {}
    disk_ini = 0.0  # no escribimos a /tmp aquí

    # ---- Body ----
    try:
        body_raw = event.get("body") or "{}"
        if event.get("isBase64Encoded"):
            body_raw = base64.b64decode(body_raw).decode("utf-8")
        body = json.loads(body_raw)
    except Exception:
        body = {}

    try:
        num_capas_ocultas = int(body.get("num_capas_ocultas", 1))
    except Exception:
        num_capas_ocultas = 1

    # ---- Validación de URLs ----
    for k, v in {
        "URL_CAPA_ENTRADA": URL_CAPA_ENTRADA,
        "URL_CAPA_OCULTA": URL_CAPA_OCULTA,
        "URL_CAPA_SALIDA": URL_CAPA_SALIDA,
        "URL_CONTENEDOR_CONFIG": URL_CONTENEDOR_CONFIG,
    }.items():
        if not v:
            return _resp(500, {"error": f"Falta variable de entorno {k}"})

    # Normaliza trailing slash
    ent = URL_CAPA_ENTRADA.rstrip("/")
    occ = URL_CAPA_OCULTA.rstrip("/")
    sal = URL_CAPA_SALIDA.rstrip("/")
    cfg = URL_CONTENEDOR_CONFIG.rstrip("/")

    # ---- Capa de entrada ----
    t1 = time.perf_counter()
    params_entrada = {"units": 5, "activation": "relu", "name": "entrada", "input_shape": [2]}
    r = requests.post(ent, json=params_entrada, timeout=60)
    tiempos["capa_entrada_s"] = round(time.perf_counter() - t1, 3)
    if r.status_code != 200:
        return _resp(500, {"error": r.text, "tiempos": tiempos})

    capas = [r.json()]  # ya es config Keras

    # ---- Capas ocultas (dinámicas) ----
    t2 = time.perf_counter()
    for i in range(num_capas_ocultas):
        t2i = time.perf_counter()
        params = {"units": 4, "activation": "relu", "name": f"oculta_{i+1}"}
        r = requests.post(occ, json=params, timeout=60)
        if r.status_code != 200:
            return _resp(500, {"error": r.text, "tiempos": tiempos})
        capas.append(r.json())
        tiempos[f"capa_oculta_{i+1}_s"] = round(time.perf_counter() - t2i, 3)
    tiempos["capas_ocultas_s"] = round(time.perf_counter() - t2, 3)

    # ---- Capa de salida ----
    t3 = time.perf_counter()
    params_sal = {"units": 1, "activation": "linear", "name": "salida"}
    r = requests.post(sal, json=params_sal, timeout=60)
    tiempos["capa_salida_s"] = round(time.perf_counter() - t3, 3)
    if r.status_code != 200:
        return _resp(500, {"error": r.text, "tiempos": tiempos})
    capas.append(r.json())

    # ---- Contenedor: crear modelo real y serializar .h5 ----
    t4 = time.perf_counter()
    payload_cfg = {"layers": capas}
    rc = requests.post(cfg, json=payload_cfg, timeout=300)
    if rc.status_code != 200:
        return _resp(500, {"error": rc.text, "tiempos": tiempos})
    tiempos["cont_config_s"] = round(time.perf_counter() - t4, 3)

    data = rc.json()
    model_b64 = data.get("model_base64")
    if not model_b64:
        return _resp(500, {"error": "El contenedor no devolvió 'model_base64'", "tiempos": tiempos})

    model_bytes = base64.b64decode(model_b64)

    # ---- Subir a S3 ----
    t5 = time.perf_counter()
    s3.put_object(Bucket=S3_BUCKET_NAME, Key=S3_KEY_H5, Body=model_bytes)
    tiempos["subir_s3_s"] = round(time.perf_counter() - t5, 3)

    # ---- Monitoreo final ----
    t_total = time.perf_counter()
    cpu_fin = proc.cpu_percent(interval=None)
    mem_fin = proc.memory_info().rss / (1024**2)
    tiempos["total_s"] = round(t_total - t0, 3)

    resp = {
        "mensaje": "Modelo creado y almacenado.",
        "tiempos": tiempos,
        "recursos": {
            "cpu_inicial_percent": cpu_ini,
            "cpu_final_percent": cpu_fin,
            "memoria_inicial_mb": round(mem_ini, 2),
            "memoria_final_mb": round(mem_fin, 2),
            "memoria_consumida_mb": round(mem_fin - mem_ini, 2),
            "disk_inicial_mb": round(disk_ini, 2),
            "disk_final_mb": 0.0,
            "disk_consumido_mb": 0.0,
            "duration_s": round(t_total - t0, 3)
        }
    }
    return _resp(200, resp)