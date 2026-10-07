import azure.functions as func
import json
import logging
import tensorflow as tf
import requests
import time
import os
import tempfile
import psutil
from azure.storage.blob import BlobServiceClient

STORAGE_CONNECTION_STRING = os.environ.get(
    "AZURE_STORAGE_CONNECTION_STRING"
)
CONTAINER_NAME = "modelos"

app = func.FunctionApp()

def limpiar_config(layer_config):
    if 'batch_input_shape' in layer_config.get("config", {}):
        del layer_config["config"]["batch_input_shape"]
    return layer_config

@app.route(route="http_trigger_modelo", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
def http_trigger_modelo(req: func.HttpRequest) -> func.HttpResponse:
    # — Monitoreo inicial —
    t0       = time.perf_counter()
    proc     = psutil.Process()
    cpu_ini  = proc.cpu_percent(interval=None)
    mem_ini  = proc.memory_info().rss / (1024**2)  # MB

    # Configuración de ruta para el archivo .h5
    tmp        = tempfile.gettempdir()
    model_name = "modelo_sin_entrenar.h5"
    model_path = os.path.join(tmp, model_name)
    # Eliminamos modelo previo si existe
    if os.path.exists(model_path):
        os.remove(model_path)
    disk_ini = 0.0

    tiempos = {}
    logging.info('Creando el modelo de red neuronal.')

    # — Parsing request —
    try:
        body = req.get_json()
        num_capas_ocultas = int(body.get("num_capas_ocultas", 1))
    except Exception:
        num_capas_ocultas = 1
    tiempos['parse_request_s'] = round(time.perf_counter() - t0, 3)

    # URLs de capas
    url_ent = "<INPUT_LAYER_FUNCTION_URL>"
    url_occ = "<HIDDEN_LAYER_FUNCTION_URL>"
    url_sal = "<OUTPUT_LAYER_FUNCTION_URL>"

    capas = []

    # — Capa de entrada —
    t1 = time.perf_counter()
    params_entrada = {"units": 5, "activation": "relu", "name": "entrada", "input_shape": [2]}
    r = requests.post(url_ent, json=params_entrada, timeout=30)
    tiempos['capa_entrada_s'] = round(time.perf_counter() - t1, 3)
    if r.status_code != 200:
        return func.HttpResponse(json.dumps({"error": r.text, "tiempos": tiempos}), status_code=500)
    layer_config = limpiar_config(r.json())
    capa_ent = tf.keras.layers.deserialize(layer_config)
    capas.append(capa_ent)

    # — Capas ocultas —
    t2 = time.perf_counter()
    for i in range(num_capas_ocultas):
        t2i = time.perf_counter()
        params = {"units": 4, "activation": "relu", "name": f"oculta_{i+1}"}
        r = requests.post(url_occ, json=params, timeout=30)
        if r.status_code != 200:
            return func.HttpResponse(json.dumps({"error": r.text, "tiempos": tiempos}), status_code=500)
        config = limpiar_config(r.json())
        capas.append(tf.keras.layers.deserialize(config))
        tiempos[f'capa_oculta_{i+1}_s'] = round(time.perf_counter() - t2i, 3)
    tiempos['capas_ocultas_s'] = round(time.perf_counter() - t2, 3)

    # — Capa de salida —
    t3 = time.perf_counter()
    params_sal = {"units": 1, "activation": "linear", "name": "salida"}
    r = requests.post(url_sal, json=params_sal, timeout=30)
    tiempos['capa_salida_s'] = round(time.perf_counter() - t3, 3)
    layer_config = limpiar_config(r.json())
    capas.append(tf.keras.layers.deserialize(layer_config))

    # — Compilación del modelo —
    t4 = time.perf_counter()
    modelo = tf.keras.Sequential(capas)
    modelo.compile(optimizer='sgd', loss='mean_squared_error', metrics=['mae', 'mse', 'mape'])
    tiempos['compilar_modelo_s'] = round(time.perf_counter() - t4, 3)

    # — Guardar modelo —
    t5 = time.perf_counter()
    modelo.save(model_path)
    tiempos['guardar_modelo_s'] = round(time.perf_counter() - t5, 3)

    # Medición de disco del modelo
    try:
        disk_fin = os.path.getsize(model_path) / (1024**2)
    except OSError:
        disk_fin = 0.0

    # — Subir a Blob —
    t6 = time.perf_counter()
    blob_client = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
    blob_client = blob_client.get_blob_client(container=CONTAINER_NAME, blob=model_name)
    with open(model_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    tiempos['subir_blob_s'] = round(time.perf_counter() - t6, 3)

    # — Monitoreo final —
    t_total = time.perf_counter()
    cpu_fin = proc.cpu_percent(interval=None)
    mem_fin = proc.memory_info().rss / (1024**2)
    tiempos['total_s'] = round(t_total - t0, 3)

    # — Respuesta JSON con métricas —
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
            "disk_final_mb": round(disk_fin, 2),
            "disk_consumido_mb": round(disk_fin - disk_ini, 2),
            "duration_s": round(t_total - t0, 3)
        }
    }
    return func.HttpResponse(json.dumps(resp, ensure_ascii=False, indent=2), status_code=200)
