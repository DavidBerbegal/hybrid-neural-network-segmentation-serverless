import azure.functions as func
import tensorflow as tf
import numpy as np
import os
import tempfile
import json
import logging
import time
import psutil
from azure.storage.blob import BlobServiceClient

# Configuración de Azure Blob Storage
STORAGE_CONNECTION_STRING = os.environ.get(
    "AZURE_STORAGE_CONNECTION_STRING"
)
CONTAINER_NAME = "modelos"
TRAINED_MODEL_BLOB = "modelo_entrenado.h5"

# Directorio temporal
temp_dir = tempfile.gettempdir()
LOCAL_TRAINED_MODEL_PATH = os.path.join(temp_dir, "modelo_entrenado.h5")

app = func.FunctionApp()

@app.route(route="http_trigger_prediccion", auth_level=func.AuthLevel.ANONYMOUS)
def http_trigger_prediccion(req: func.HttpRequest) -> func.HttpResponse:
    tiempos = {}
    t_total_inicio = time.perf_counter()
    process = psutil.Process(os.getpid())

    # MONITORIZACIÓN INICIAL
    memoria_inicial = process.memory_info().rss / 1024**2
    cpu_inicial = process.cpu_percent(interval=None)
    logging.info(f"[MONITOR] Memoria inicial (MB): {memoria_inicial:.2f}")
    logging.info(f"[MONITOR] CPU inicial (%): {cpu_inicial}")

    # Paso 1: Descargar modelo
    t1 = time.perf_counter()
    try:
        blob_service_client = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=TRAINED_MODEL_BLOB)
        with open(LOCAL_TRAINED_MODEL_PATH, "wb") as f:
            data = blob_client.download_blob()
            f.write(data.readall())
        tiempos["descargar_modelo"] = round(time.perf_counter() - t1, 3)
        logging.info("Modelo descargado correctamente.")
    except Exception as e:
        msg = f"Error al descargar el modelo entrenado: {str(e)}"
        tiempos["descargar_modelo"] = round(time.perf_counter() - t1, 3)
        return func.HttpResponse(
            json.dumps({"error": msg, "tiempos": tiempos}, indent=2),
            status_code=500,
            mimetype="application/json"
        )

    # Paso 2: Cargar modelo
    t1 = time.perf_counter()
    try:
        modelo_cargado = tf.keras.models.load_model(LOCAL_TRAINED_MODEL_PATH)
        tiempos["cargar_modelo"] = round(time.perf_counter() - t1, 3)
        logging.info("Modelo cargado correctamente.")
    except Exception as e:
        msg = f"Error al cargar el modelo entrenado: {str(e)}"
        tiempos["cargar_modelo"] = round(time.perf_counter() - t1, 3)
        return func.HttpResponse(
            json.dumps({"error": msg, "tiempos": tiempos}, indent=2),
            status_code=500,
            mimetype="application/json"
        )

    # Paso 3: Procesar entrada
    t1 = time.perf_counter()
    try:
        req_body = req.get_json()
        nueva_entrada = np.array(req_body["entrada"], dtype=float)
        if len(nueva_entrada.shape) == 1:
            nueva_entrada = nueva_entrada.reshape(1, -1)
        tiempos["procesar_entrada"] = round(time.perf_counter() - t1, 3)
        logging.info(f"Entrada procesada correctamente: {nueva_entrada.shape}")
    except Exception as e:
        msg = f"Error al procesar la entrada: {str(e)}"
        tiempos["procesar_entrada"] = round(time.perf_counter() - t1, 3)
        return func.HttpResponse(
            json.dumps({"error": msg, "tiempos": tiempos}, indent=2),
            status_code=400,
            mimetype="application/json"
        )

    # Paso 4: Predicción
    t1 = time.perf_counter()
    try:
        resultado = modelo_cargado.predict(nueva_entrada)
        valor_predicho = float(resultado[0][0])
        tiempos["predecir"] = round(time.perf_counter() - t1, 3)
        logging.info(f"Predicción generada correctamente: {valor_predicho:.4f}")
    except Exception as e:
        msg = f"Error durante la predicción: {str(e)}"
        tiempos["predecir"] = round(time.perf_counter() - t1, 3)
        return func.HttpResponse(
            json.dumps({"error": msg, "tiempos": tiempos}, indent=2),
            status_code=500,
            mimetype="application/json"
        )

    # MONITORIZACIÓN FINAL
    memoria_final = process.memory_info().rss / 1024**2
    cpu_final = process.cpu_percent(interval=None)
    tiempos["memoria_inicial_mb"] = round(memoria_inicial, 2)
    tiempos["memoria_final_mb"] = round(memoria_final, 2)
    tiempos["cpu_inicial_percent"] = cpu_inicial
    tiempos["cpu_final_percent"] = cpu_final
    tiempos["total"] = round(time.perf_counter() - t_total_inicio, 3)

    return func.HttpResponse(
        json.dumps({
            "prediccion": valor_predicho,
            "tiempos": tiempos
        }, ensure_ascii=False, indent=2),
        mimetype="application/json",
        status_code=200
    )
