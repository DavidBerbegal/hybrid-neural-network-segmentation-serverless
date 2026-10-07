import azure.functions as func
import tensorflow as tf
import numpy as np
import logging
import os
import tempfile
import json
import requests
import time
import psutil
import pandas as pd
from azure.storage.blob import BlobServiceClient

STORAGE_CONNECTION_STRING = os.environ.get(
    "AZURE_STORAGE_CONNECTION_STRING"
)
CONTAINER_NAME = "modelos"
UNTRAINED_MODEL_BLOB = "modelo_sin_entrenar.h5"
TRAINED_MODEL_BLOB = "modelo_entrenado.h5"
DATOS_CSV_BLOB = "datos_entrenamiento.csv"

temp_dir = tempfile.gettempdir()
LOCAL_UNTRAINED_MODEL_PATH = os.path.join(temp_dir, "modelo_sin_entrenar.h5")
LOCAL_TRAINED_MODEL_PATH   = os.path.join(temp_dir, "modelo_entrenado.h5")
LOCAL_CSV_PATH             = os.path.join(temp_dir, "datos_entrenamiento.csv")

def uso_disco_temporal():
    total = 0
    for root, _, files in os.walk(tempfile.gettempdir()):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except:
                pass
    return total/(1024**2)

app = func.FunctionApp()

@app.route(route="http_trigger_entrenamiento", auth_level=func.AuthLevel.ANONYMOUS)
def http_trigger_entrenamiento(req: func.HttpRequest) -> func.HttpResponse:
    tiempos = {}
    start_total = time.perf_counter()
    proc = psutil.Process(os.getpid())

    # Inicial recursos
    cpu_ini = proc.cpu_percent(interval=None)
    mem_ini = proc.memory_info().rss/(1024**2)
    disk_ini = uso_disco_temporal()

    # Parámetros
    body = req.get_json() if req.method!="GET" else req.params
    total_epochs = int(body.get("total_epochs"))
    epoch_inicio = int(body.get("epoch_inicio",0))
    block_size   = int(body.get("block_size"))
    batch_size   = int(body.get("batch_size"))
    epoch_final  = min(epoch_inicio+block_size, total_epochs)

    # Descargar modelo
    t = time.perf_counter()
    bs = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
    client = bs.get_blob_client(CONTAINER_NAME, TRAINED_MODEL_BLOB)
    try:
        with open(LOCAL_UNTRAINED_MODEL_PATH,'wb') as f:
            f.write(client.download_blob().readall())
    except:
        client = bs.get_blob_client(CONTAINER_NAME, UNTRAINED_MODEL_BLOB)
        with open(LOCAL_UNTRAINED_MODEL_PATH,'wb') as f:
            f.write(client.download_blob().readall())
    tiempos['descargar_modelo_s'] = round(time.perf_counter()-t,3)

    # Descargar CSV
    t = time.perf_counter()
    client = bs.get_blob_client(CONTAINER_NAME, DATOS_CSV_BLOB)
    with open(LOCAL_CSV_PATH,'wb') as f:
        f.write(client.download_blob().readall())
    tiempos['descargar_csv_s'] = round(time.perf_counter()-t,3)

    # Cargar datos
    t = time.perf_counter()
    df = pd.read_csv(LOCAL_CSV_PATH, nrows=batch_size)
    X = df.iloc[:,:2].astype(np.float32).values
    Y = df.iloc[:,2].astype(np.float32).values.reshape(-1,1)
    tiempos['cargar_csv_s'] = round(time.perf_counter()-t,3)

    # Cargar y compilar modelo
    t = time.perf_counter()
    model = tf.keras.models.load_model(LOCAL_UNTRAINED_MODEL_PATH)
    model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['mae','mse','mape'])
    tiempos['compilar_modelo_s'] = round(time.perf_counter()-t,3)

    # Entrenamiento
    t = time.perf_counter()
    history = model.fit(X,Y, epochs=epoch_final, initial_epoch=epoch_inicio, verbose=0)
    tiempos['entrenamiento_s'] = round(time.perf_counter()-t,3)

    # Guardar modelo
    t = time.perf_counter()
    model.save(LOCAL_TRAINED_MODEL_PATH)
    tiempos['guardar_modelo_s'] = round(time.perf_counter()-t,3)

    # Subir modelo
    t = time.perf_counter()
    client = bs.get_blob_client(CONTAINER_NAME, TRAINED_MODEL_BLOB)
    with open(LOCAL_TRAINED_MODEL_PATH,'rb') as f:
        client.upload_blob(f, overwrite=True)
    tiempos['subir_modelo_s'] = round(time.perf_counter()-t,3)

    # Final recursos
    cpu_fin = proc.cpu_percent(interval=None)
    mem_fin = proc.memory_info().rss/(1024**2)
    disk_fin = uso_disco_temporal()
    total_time = round(time.perf_counter()-start_total,3)

    tiempos.update({
        'cpu_inicial_percent': cpu_ini,
        'cpu_final_percent': cpu_fin,
        'memoria_inicial_mb': round(mem_ini,2),
        'memoria_final_mb': round(mem_fin,2),
        'memoria_consumida_mb': round(mem_fin-mem_ini,2),
        'disk_inicial_mb': round(disk_ini,2),
        'disk_final_mb': round(disk_fin,2),
        'disk_consumido_mb': round(disk_fin-disk_ini,2),
        'total_s': total_time
    })

    return func.HttpResponse(
        json.dumps({
            'bloque_completado':[epoch_inicio,epoch_final],
            'epocas_totales':total_epochs,
            'tiempos':tiempos,
            'metricas_finales':{k:float(history.history[k][-1]) for k in history.history},
            'registros_totales':int(X.shape[0])
        }, ensure_ascii=False, indent=2),
        status_code=200,
        mimetype='application/json'
    )
