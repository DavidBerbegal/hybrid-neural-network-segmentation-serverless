import azure.functions as func
import numpy as np
import os
import tempfile
import csv
import time
import json
import psutil
from azure.storage.blob import BlobServiceClient

# Configuración de Azure Blob Storage
STORAGE_CONNECTION_STRING = os.environ.get(
    "STORAGE_CONNECTION_STRING"
)

CONTAINER_NAME = "modelos"

app = func.FunctionApp()

@app.route(route="http_trigger_datos", auth_level=func.AuthLevel.ANONYMOUS, methods=["GET","POST"])
def http_trigger_datos(req: func.HttpRequest) -> func.HttpResponse:
    # — Monitoreo inicial —
    t0 = time.perf_counter()
    proc = psutil.Process()
    cpu_ini = proc.cpu_percent(interval=None)
    mem_ini = proc.memory_info().rss / (1024**2)  # MB

    # Ruta al CSV temporal y limpieza de versión previa
    tmp = tempfile.gettempdir()
    csv_name = "datos_entrenamiento.csv"
    path = os.path.join(tmp, csv_name)
    if os.path.exists(path):
        os.remove(path)
    disk_ini = 0.0

    tiempos = {}

    # Parámetros de entrada
    try:
        if req.method == "POST":
            body = req.get_json()
            n = int(body.get("n", 1000))
        else:
            n = int(req.params.get("n", 1000))
    except:
        n = 1000
    offset = max(0, int(req.params.get("offset", 0)))
    limit = max(1, int(req.params.get("limit", n)))
    end = min(offset + limit, n)
    tiempos['parse_params_s'] = round(time.perf_counter() - t0, 3)

    # Generación de pares de datos
    t1 = time.perf_counter()
    np.random.seed(42)
    X = np.random.uniform(0, 10, (n, 2))
    Y = X.sum(axis=1).reshape(-1, 1)
    X_sub, Y_sub = X[offset:end], Y[offset:end]
    tiempos['generate_data_s'] = round(time.perf_counter() - t1, 3)

    # Escritura local del CSV
    t2 = time.perf_counter()
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x1", "x2", "y"])
        for x, y in zip(X_sub, Y_sub):
            writer.writerow([x[0], x[1], y[0]])
    tiempos['write_csv_s'] = round(time.perf_counter() - t2, 3)

    # Medición de disco del CSV generado
    try:
        disk_fin = os.path.getsize(path) / (1024**2)
    except OSError:
        disk_fin = 0.0

    # Subida a Blob Storage
    t3 = time.perf_counter()
    try:
        svc = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
        blob = svc.get_blob_client(container=CONTAINER_NAME, blob=csv_name)
        with open(path, "rb") as data:
            blob.upload_blob(data, overwrite=True)
        tiempos['upload_blob_s'] = round(time.perf_counter() - t3, 3)
    except Exception as e:
        tiempos['upload_blob_s'] = round(time.perf_counter() - t3, 3)
        return func.HttpResponse(
            json.dumps({
                "status": "error",
                "mensaje": f"Error subiendo CSV al blob: {e}",
                "tiempos": tiempos
            }, ensure_ascii=False, indent=2),
            status_code=500,
            mimetype="application/json"
        )

    # Monitoreo final
    t_total = time.perf_counter()
    cpu_fin = proc.cpu_percent(interval=None)
    mem_fin = proc.memory_info().rss / (1024**2)
    tiempos['total_s'] = round(t_total - t0, 3)

    # Construcción de la respuesta JSON
    resp = {
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
    }
    return func.HttpResponse(
        json.dumps(resp, ensure_ascii=False, indent=2),
        status_code=200,
        mimetype="application/json"
    )
