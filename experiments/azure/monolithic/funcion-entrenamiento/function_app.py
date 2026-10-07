import azure.functions as func
import json
import logging
import numpy as np
import tensorflow as tf
import os
import tempfile
import time
import psutil
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError
from keras.layers import Dense

# Configuración de Azure Blob Storage
STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "modelos"
TRAINED_MODEL_BLOB = "modelo_entrenado.h5"

app = func.FunctionApp()

def obtener_datos(n=1000):
    np.random.seed(42)
    X = np.random.uniform(0, 10, (n, 2))
    Y = (X[:, 0] + X[:, 1]).reshape(-1, 1)
    return X, Y

def crear_capa_entrada(units, input_shape, activation, name="capa_entrada_dense"):
    return Dense(units=units, activation=activation, input_shape=tuple(input_shape), name=name)

def crear_capa_oculta(units, activation, name):
    return Dense(units=units, activation=activation, name=name)

def crear_capa_salida(units=1, activation="linear", name="capa_salida_dense"):
    return Dense(units=units, activation=activation, name=name)

def generar_modelo(num_capas_ocultas):
    capas = [crear_capa_entrada(units=5, activation="relu", input_shape=[2], name="entrada")]
    for i in range(num_capas_ocultas):
        capas.append(crear_capa_oculta(units=4, activation="relu", name=f"oculta_{i+1}"))
    capas.append(crear_capa_salida(units=1, activation="linear", name="salida"))
    return tf.keras.Sequential(capas)

def guardar_en_blob(local_path, blob_name):
    blob_service_client = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)
    with open(local_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)

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

@app.route(route="http_trigger_entrenamiento_monolitica", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
def http_trigger_entrenamiento_monolitica(req: func.HttpRequest) -> func.HttpResponse:
    tiempos = {}
    entrenamiento_batch = []
    metricas = {}
    t_total_inicio = time.perf_counter()
    process = psutil.Process(os.getpid())

    # MONITORIZACIÓN INICIAL
    memoria_inicial = process.memory_info().rss / 1024**2
    disco_inicial = uso_disco_temporal()
    cpu_inicial = process.cpu_percent(interval=None)

    # PARÁMETROS
    try:
        body = req.get_json()
        total_epochs      = int(body["total_epochs"])
        epoch_inicio      = int(body.get("epoch_inicio", 0))
        block_size        = int(body.get("block_size", 100))
        num_capas_ocultas = int(body.get("num_capas_ocultas", 1))
        n_datos           = int(body.get("n_datos", 1000))
    except Exception as e:
        return func.HttpResponse(
            json.dumps({"error": "Error en parámetros de entrada: " + str(e)}),
            status_code=400,
            mimetype="application/json"
        )

    epoch_final = min(epoch_inicio + block_size, total_epochs)

    # GENERACIÓN DE DATOS
    t1 = time.perf_counter()
    X, Y = obtener_datos(n_datos)
    tiempos["generacion_datos_s"] = round(time.perf_counter() - t1, 3)

    # CREACIÓN Y COMPILACIÓN DE MODELO
    t1 = time.perf_counter()
    modelo = generar_modelo(num_capas_ocultas)
    modelo.compile(
        optimizer="sgd",
        loss="mean_squared_error",
        metrics=["mae", "mse", "mape"]
    )
    tiempos["compilacion_modelo_s"] = round(time.perf_counter() - t1, 3)

    # ENTRENAMIENTO
    t1 = time.perf_counter()
    history = modelo.fit(
        X, Y,
        epochs=epoch_final,
        initial_epoch=epoch_inicio,
        verbose=0
    )
    tiempos["entrenamiento_s"] = round(time.perf_counter() - t1, 3)

    # METRICS
    batch_metrics = {}
    for key in ["loss", "mae", "mse", "mape"]:
        if key in history.history:
            batch_metrics[key] = float(history.history[key][-1])
            metricas[key] = batch_metrics[key]

    entrenamiento_batch.append({
        "batch": 0,
        "registros": int(X.shape[0]),
        "metricas": batch_metrics,
        "memoria_pre_batch_mb": round(memoria_inicial, 2),
        "memoria_post_batch_mb": round(process.memory_info().rss / 1024**2, 2),
        "cpu_pre_batch_percent": cpu_inicial,
        "cpu_post_batch_percent": process.cpu_percent(interval=None),
        "tiempo_batch_s": round(time.perf_counter() - t1, 2)
    })

    # GUARDADO Y SUBIDA
    t1 = time.perf_counter()
    temp_path = os.path.join(tempfile.gettempdir(), "modelo_entrenado.h5")
    modelo.save(temp_path)
    guardar_en_blob(temp_path, TRAINED_MODEL_BLOB)
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

    return func.HttpResponse(
        json.dumps({
            "bloque_completado": [epoch_inicio, epoch_final],
            "epocas_totales": total_epochs,
            "tiempos": tiempos,
            "metricas_finales": metricas,
            "entrenamiento_batch": entrenamiento_batch,
            "registros_totales": int(X.shape[0])
        }, ensure_ascii=False, indent=2),
        status_code=200,
        mimetype="application/json"
    )
