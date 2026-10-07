import os, json, time, base64, logging, tempfile, csv
import boto3
import requests
import numpy as np
import psutil

logging.basicConfig(level=logging.INFO)

# ENV
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "red-neuronal-segmentada-modelos")
UNTRAINED_KEY  = os.environ.get("UNTRAINED_KEY",  "modelo_sin_entrenar.h5")
TRAINED_KEY    = os.environ.get("TRAINED_KEY",    "modelo_entrenado.h5")
DATOS_KEY      = os.environ.get("DATOS_KEY",      "datos_entrenamiento.csv")
CONTAINER_TRAIN_URL = os.environ.get("CONTAINER_TRAIN_URL", "")  # .../train-model

s3 = boto3.client("s3")

def _resp(code, body):
    if isinstance(body, (dict, list)):
        body = json.dumps(body, ensure_ascii=False, indent=2)
    return {"statusCode": code, "headers": {"Content-Type":"application/json"}, "body": body}

def _method(event):
    return (event.get("httpMethod")
            or event.get("requestContext", {}).get("http", {}).get("method")
            or "GET").upper()

def _uso_tmp_mb():
    total = 0
    tmp = tempfile.gettempdir()
    for root, _, files in os.walk(tmp):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except:
                pass
    return total/(1024**2)

def lambda_handler(event, context):
    if _method(event) == "GET":
        return _resp(200, "Función entrenamiento activa (AWS orquesta; contenedor entrena)")

    # ---- Parse body ----
    try:
        body_raw = event.get("body") or "{}"
        if event.get("isBase64Encoded"):
            body_raw = base64.b64decode(body_raw).decode("utf-8")
        body = json.loads(body_raw)
    except Exception:
        body = {}

    try:
        total_epochs = int(body.get("total_epochs"))
        epoch_inicio = int(body.get("epoch_inicio", 0))
        block_size   = int(body.get("block_size"))
        batch_size   = int(body.get("batch_size"))
    except Exception as e:
        return _resp(400, {"error": f"Parámetros inválidos: {e}"})

    epoch_final  = min(epoch_inicio + block_size, total_epochs)

    # ---- Métricas inicio ----
    tiempos = {}
    t_total = time.perf_counter()
    proc = psutil.Process(os.getpid())
    cpu_ini = proc.cpu_percent(interval=None)
    mem_ini = proc.memory_info().rss/(1024**2)
    disk_ini = _uso_tmp_mb()

    # ---- Descarga modelo (entrenado si existe; si no, sin entrenar) ----
    tmp = tempfile.gettempdir()
    local_untrained = os.path.join(tmp, "modelo_base.h5")

    t = time.perf_counter()
    try:
        s3.download_file(S3_BUCKET_NAME, TRAINED_KEY, local_untrained)
    except Exception:
        s3.download_file(S3_BUCKET_NAME, UNTRAINED_KEY, local_untrained)
    tiempos["descargar_modelo_s"] = round(time.perf_counter() - t, 3)

    # ---- Descarga CSV ----
    local_csv = os.path.join(tmp, "datos_entrenamiento.csv")
    t = time.perf_counter()
    s3.download_file(S3_BUCKET_NAME, DATOS_KEY, local_csv)
    tiempos["descargar_csv_s"] = round(time.perf_counter() - t, 3)

    # ---- Cargar primeros batch_size registros (x1,x2,y) ----
    t = time.perf_counter()
    X_list, Y_list = [], []
    with open(local_csv, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # salta cabecera
        for i, row in enumerate(reader):
            if i >= batch_size:
                break
            x1, x2, y = float(row[0]), float(row[1]), float(row[2])
            X_list.append([x1, x2])
            Y_list.append([y])
    X = np.array(X_list, dtype=float)
    Y = np.array(Y_list, dtype=float)
    tiempos["cargar_csv_s"] = round(time.perf_counter() - t, 3)

    # ---- Enviar a contenedor para entrenar ----
    with open(local_untrained, "rb") as f:
        model_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model_base64": model_b64,
        "X": X.tolist(),
        "Y": Y.tolist(),
        # paridad con Azure
        "optimizer": "sgd",
        "loss": "mean_squared_error",
        "metrics": ["mae", "mse", "mape"],
        "epochs": epoch_final,
        "initial_epoch": epoch_inicio
    }

    # En el script, tiempo_preparacion = descargar_csv_s + compilar_modelo_s
    # Aquí la "compilación" vive en el contenedor; si no la reporta, devolvemos 0.
    tiempos["compilar_modelo_s"] = 0.0

    t = time.perf_counter()
    r = requests.post(CONTAINER_TRAIN_URL, json=payload, timeout=900)
    tiempos["entrenamiento_s"] = round(time.perf_counter() - t, 3)
    if r.status_code != 200:
        # Aun en error, devolvemos recursos/tiempos para diagnosticar
        cpu_fin = proc.cpu_percent(interval=None)
        mem_fin = proc.memory_info().rss/(1024**2)
        disk_fin = _uso_tmp_mb()
        tiempos.update({
            "cpu_inicial_percent": cpu_ini,
            "cpu_final_percent":   cpu_fin,
            "memoria_inicial_mb":  round(mem_ini, 2),
            "memoria_final_mb":    round(mem_fin, 2),
            "memoria_consumida_mb": round(mem_fin - mem_ini, 2),
            "disk_inicial_mb":     round(disk_ini, 2),
            "disk_final_mb":       round(disk_fin, 2),
            "disk_consumido_mb":   round(disk_fin - disk_ini, 2),
            "total_s":             round(time.perf_counter() - t_total, 3)
        })
        return _resp(500, {"error": r.text, "tiempos": tiempos})

    out = r.json()
    trained_b64 = out.get("model_base64")
    # El contenedor ya devuelve métricas coherentes:
    metricas_finales = out.get("metrics") or out.get("history_last") or {}
    if not trained_b64:
        return _resp(500, {"error": "El contenedor no devolvió 'model_base64'", "tiempos": tiempos})
    trained_bytes = base64.b64decode(trained_b64)

    # ---- Guardar modelo entrenado en S3 ----
    t = time.perf_counter()
    s3.put_object(Bucket=S3_BUCKET_NAME, Key=TRAINED_KEY, Body=trained_bytes)
    tiempos["subir_modelo_s"] = round(time.perf_counter() - t, 3)

    # ---- Final métricas (A PLANO EN 'tiempos', como espera tu CSV) ----
    cpu_fin = proc.cpu_percent(interval=None)
    mem_fin = proc.memory_info().rss/(1024**2)
    disk_fin = _uso_tmp_mb()
    tiempos.update({
        "cpu_inicial_percent": cpu_ini,
        "cpu_final_percent":   cpu_fin,
        "memoria_inicial_mb":  round(mem_ini, 2),
        "memoria_final_mb":    round(mem_fin, 2),
        "memoria_consumida_mb": round(mem_fin - mem_ini, 2),
        "disk_inicial_mb":     round(disk_ini, 2),
        "disk_final_mb":       round(disk_fin, 2),
        "disk_consumido_mb":   round(disk_fin - disk_ini, 2),
        "total_s":             round(time.perf_counter() - t_total, 3)
    })

    # ---- Respuesta con el MISMO formato que Azure ----
    return _resp(200, {
        "bloque_completado": [epoch_inicio, epoch_final],
        "epocas_totales": total_epochs,
        "tiempos": tiempos,
        "metricas_finales": metricas_finales,
        "registros_totales": int(X.shape[0])
    })
