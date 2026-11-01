import base64
import csv
import json
import os
import tempfile
import time

import boto3
import numpy as np
import psutil


S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
CSV_NAME = "datos_entrenamiento.csv"

s3 = boto3.client("s3")


def response(status_code, payload):
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload, ensure_ascii=False, indent=2),
    }


def lambda_handler(event, context):
    # Monitorización inicial, con el mismo alcance que Azure.
    t0 = time.perf_counter()
    proc = psutil.Process()
    cpu_ini = proc.cpu_percent(interval=None)
    mem_ini = proc.memory_info().rss / (1024 ** 2)

    tmp = tempfile.gettempdir()
    path = os.path.join(tmp, CSV_NAME)
    if os.path.exists(path):
        os.remove(path)
    disk_ini = 0.0

    tiempos = {}

    method = (
        event.get("httpMethod")
        or event.get("requestContext", {}).get("http", {}).get("method")
        or "GET"
    ).upper()

    if method not in ("GET", "POST"):
        return response(405, {"error": "Método no soportado"})

    params = event.get("queryStringParameters") or {}

    # Igual que Azure: POST obtiene n del JSON; GET, de la URL.
    try:
        if method == "POST":
            body_raw = event.get("body") or "{}"
            if event.get("isBase64Encoded"):
                body_raw = base64.b64decode(body_raw).decode("utf-8")
            body = json.loads(body_raw)
            n = int(body.get("n", 1000))
        else:
            n = int(params.get("n", 1000))
    except Exception:
        n = 1000

    # Igual que Azure: offset y limit siempre proceden de la URL.
    offset = max(0, int(params.get("offset", 0)))
    limit = max(1, int(params.get("limit", n)))
    end = min(offset + limit, n)
    tiempos["parse_params_s"] = round(time.perf_counter() - t0, 3)

    # Generación de datos.
    t1 = time.perf_counter()
    np.random.seed(42)
    X = np.random.uniform(0, 10, (n, 2))
    Y = X.sum(axis=1).reshape(-1, 1)
    X_sub, Y_sub = X[offset:end], Y[offset:end]
    tiempos["generate_data_s"] = round(time.perf_counter() - t1, 3)

    # Escritura local del CSV.
    t2 = time.perf_counter()
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x1", "x2", "y"])
        for x, y in zip(X_sub, Y_sub):
            writer.writerow([x[0], x[1], y[0]])
    tiempos["write_csv_s"] = round(time.perf_counter() - t2, 3)

    try:
        disk_fin = os.path.getsize(path) / (1024 ** 2)
    except OSError:
        disk_fin = 0.0

    # Adaptación de Blob Storage a S3.
    t3 = time.perf_counter()
    try:
        if not S3_BUCKET_NAME:
            raise ValueError("Falta la variable de entorno S3_BUCKET_NAME.")

        with open(path, "rb") as data:
            s3.upload_fileobj(data, S3_BUCKET_NAME, CSV_NAME)

        tiempos["upload_s3_s"] = round(time.perf_counter() - t3, 3)
    except Exception as e:
        tiempos["upload_s3_s"] = round(time.perf_counter() - t3, 3)
        return response(
            500,
            {
                "status": "error",
                "mensaje": f"Error subiendo CSV a S3: {e}",
                "tiempos": tiempos,
            },
        )

    # Monitorización final.
    t_total = time.perf_counter()
    cpu_fin = proc.cpu_percent(interval=None)
    mem_fin = proc.memory_info().rss / (1024 ** 2)
    tiempos["total_s"] = round(t_total - t0, 3)

    return response(
        200,
        {
            "status": "success",
            "mensaje": "CSV generado y subido correctamente.",
            "resultado": {"n": n, "offset": offset, "limit": limit},
            "tiempos": tiempos,
            "recursos": {
                "cpu_inicial_percent": cpu_ini,
                "cpu_final_percent": cpu_fin,
                "memoria_inicial_mb": round(mem_ini, 2),
                "memoria_final_mb": round(mem_fin, 2),
                "memoria_consumida_mb": round(mem_fin - mem_ini, 2),
                "disk_inicial_mb": round(disk_ini, 2),
                "disk_final_mb": round(disk_fin, 2),
                "disk_consumido_mb": round(disk_fin - disk_ini, 2),
                "duration_s": round(t_total - t0, 3),
            },
        },
    )