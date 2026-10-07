import json
import os
import tempfile
import time
import csv
import psutil
import boto3
import numpy as np

S3_BUCKET_NAME = "red-neuronal-segmentada-modelos"

s3 = boto3.client("s3")

def lambda_handler(event, context):
    # Monitoreo inicial
    t0 = time.perf_counter()
    proc = psutil.Process()
    cpu_ini = proc.cpu_percent(interval=None)
    mem_ini = proc.memory_info().rss / (1024**2)  # MB
    disk_ini = 0.0
    tiempos = {}

    # Parámetros de entrada
    try:
        if event.get("httpMethod") == "POST":
            body = json.loads(event.get("body") or "{}")
            n = int(body.get("n", 1000))
            offset = max(0, int(body.get("offset", 0)))
            limit = max(1, int(body.get("limit", n)))
        else:
            params = event.get("queryStringParameters") or {}
            n = int(params.get("n", 1000))
            offset = max(0, int(params.get("offset", 0)))
            limit = max(1, int(params.get("limit", n)))
    except Exception:
        n = 1000
        offset = 0
        limit = n
    end = min(offset + limit, n)
    tiempos['parse_params_s'] = round(time.perf_counter() - t0, 3)

    # Generación de datos
    t1 = time.perf_counter()
    np.random.seed(42)
    X = np.random.uniform(0, 10, (n, 2))
    Y = X.sum(axis=1).reshape(-1, 1)
    X_sub, Y_sub = X[offset:end], Y[offset:end]
    tiempos['generate_data_s'] = round(time.perf_counter() - t1, 3)

    # Escritura a CSV
    t2 = time.perf_counter()
    tmp_dir = tempfile.gettempdir()
    csv_name = "datos_entrenamiento.csv"
    path = os.path.join(tmp_dir, csv_name)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x1", "x2", "y"])
        for x, y in zip(X_sub, Y_sub):
            writer.writerow([x[0], x[1], y[0]])
    tiempos['write_csv_s'] = round(time.perf_counter() - t2, 3)

    # Tamaño en disco
    try:
        disk_fin = os.path.getsize(path) / (1024**2)
    except:
        disk_fin = 0.0

    # Subida a S3
    t3 = time.perf_counter()
    try:
        with open(path, "rb") as data:
            s3.upload_fileobj(data, S3_BUCKET_NAME, csv_name)
        tiempos['upload_s3_s'] = round(time.perf_counter() - t3, 3)
    except Exception as e:
        tiempos['upload_s3_s'] = round(time.perf_counter() - t3, 3)
        return response(500, {
            "status": "error",
            "mensaje": f"Error subiendo a S3: {str(e)}",
            "tiempos": tiempos
        })

    # Monitoreo final
    t_total = time.perf_counter()
    cpu_fin = proc.cpu_percent(interval=None)
    mem_fin = proc.memory_info().rss / (1024**2)

    return response(200, {
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
            "duration_s": round(t_total - t0, 3)
        }
    })

def response(status_code, body_dict):
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body_dict, ensure_ascii=False, indent=2)
    }
