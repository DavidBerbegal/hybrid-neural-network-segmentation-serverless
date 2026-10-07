import azure.functions as func
import json
import logging
import os
import tempfile
import time

app = func.FunctionApp()

# App Settings (Azure) / local.settings.json
STORAGE_CONNECTION_STRING = os.getenv(
    "STORAGE_CONNECTION_STRING"
)

CONTAINER_NAME = os.getenv("MODEL_CONTAINER_NAME", "modelos")
UNTRAINED_MODEL_BLOB = os.getenv("UNTRAINED_MODEL_BLOB", "modelo_sin_entrenar.h5")
TRAINED_MODEL_BLOB = os.getenv("TRAINED_MODEL_BLOB", "modelo_entrenado.h5")
DATOS_CSV_BLOB = os.getenv("DATOS_CSV_BLOB", "datos_entrenamiento.csv")

temp_dir = tempfile.gettempdir()
LOCAL_UNTRAINED_MODEL_PATH = os.path.join(temp_dir, "modelo_sin_entrenar.h5")
LOCAL_TRAINED_MODEL_PATH = os.path.join(temp_dir, "modelo_entrenado.h5")
LOCAL_CSV_PATH = os.path.join(temp_dir, "datos_entrenamiento.csv")


def uso_disco_temporal() -> float:
    total = 0
    for root, _, files in os.walk(tempfile.gettempdir()):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except Exception:
                pass
    return total / (1024 ** 2)


def _get_body_params(req: func.HttpRequest) -> dict:
    # Mantiene tu idea: GET -> params, POST -> json
    if req.method == "GET":
        return dict(req.params)
    try:
        return req.get_json()
    except Exception:
        return {}


@app.route(
    route="http_trigger_entrenamiento",
    auth_level=func.AuthLevel.ANONYMOUS,
    methods=["GET", "POST"],
)
def http_trigger_entrenamiento(req: func.HttpRequest) -> func.HttpResponse:
    # Imports pesados DENTRO -> no rompen el indexing del host
    import numpy as np
    import pandas as pd
    import psutil
    import tensorflow as tf
    from azure.storage.blob import BlobServiceClient

    tiempos = {}
    start_total = time.perf_counter()
    proc = psutil.Process(os.getpid())

    # Recursos iniciales
    cpu_ini = proc.cpu_percent(interval=None)
    mem_ini = proc.memory_info().rss / (1024 ** 2)
    disk_ini = uso_disco_temporal()

    # Validar settings mínimos
    if not STORAGE_CONNECTION_STRING:
        return func.HttpResponse(
            json.dumps({"error": "Falta STORAGE_CONNECTION_STRING en App Settings."}, ensure_ascii=False),
            status_code=500,
            mimetype="application/json",
        )

    # Parámetros (mismos que tu código)
    body = _get_body_params(req)
    try:
        total_epochs = int(body.get("total_epochs"))
        epoch_inicio = int(body.get("epoch_inicio", 0))
        block_size = int(body.get("block_size"))
        batch_size = int(body.get("batch_size"))
    except Exception as e:
        return func.HttpResponse(
            json.dumps(
                {
                    "error": "Parámetros inválidos. Obligatorios: total_epochs, block_size, batch_size. Opcional: epoch_inicio.",
                    "detalle": str(e),
                    "body": body,
                },
                ensure_ascii=False,
                indent=2,
            ),
            status_code=400,
            mimetype="application/json",
        )

    epoch_final = min(epoch_inicio + block_size, total_epochs)

    # Asegura limpieza de ficheros previos (evita “cosas viejas”)
    for p in (LOCAL_UNTRAINED_MODEL_PATH, LOCAL_TRAINED_MODEL_PATH, LOCAL_CSV_PATH):
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

    logging.info(
        f"Entrenamiento: epochs [{epoch_inicio}, {epoch_final}] de total {total_epochs}, batch_size={batch_size}"
    )

    # Cliente Blob
    bs = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)

    # Descargar modelo (primero entrenado si existe, sino sin entrenar)
    t = time.perf_counter()
    try:
        client = bs.get_blob_client(CONTAINER_NAME, TRAINED_MODEL_BLOB)
        with open(LOCAL_UNTRAINED_MODEL_PATH, "wb") as f:
            f.write(client.download_blob().readall())
        modelo_origen = TRAINED_MODEL_BLOB
    except Exception:
        client = bs.get_blob_client(CONTAINER_NAME, UNTRAINED_MODEL_BLOB)
        with open(LOCAL_UNTRAINED_MODEL_PATH, "wb") as f:
            f.write(client.download_blob().readall())
        modelo_origen = UNTRAINED_MODEL_BLOB
    tiempos["descargar_modelo_s"] = round(time.perf_counter() - t, 3)

    # Descargar CSV
    t = time.perf_counter()
    client = bs.get_blob_client(CONTAINER_NAME, DATOS_CSV_BLOB)
    with open(LOCAL_CSV_PATH, "wb") as f:
        f.write(client.download_blob().readall())
    tiempos["descargar_csv_s"] = round(time.perf_counter() - t, 3)

    # Cargar datos (nrows=batch_size, X 2 cols, Y 3ª col)
    t = time.perf_counter()
    df = pd.read_csv(LOCAL_CSV_PATH, nrows=batch_size)
    X = df.iloc[:, :2].astype(np.float32).values
    Y = df.iloc[:, 2].astype(np.float32).values.reshape(-1, 1)
    tiempos["cargar_csv_s"] = round(time.perf_counter() - t, 3)

    # ─────────────────────────────────────────────────────────────
    # Cargar y compilar modelo (AHORA con soporte SplitDense)
    # ─────────────────────────────────────────────────────────────
    t = time.perf_counter()

    @tf.keras.utils.register_keras_serializable(package="custom")
    class SplitDense(tf.keras.layers.Layer):
        def __init__(self, parts, axis=-1, split_mode="units", recombine="concat",
                     units_total=None, name=None, **kwargs):
            super().__init__(name=name, **kwargs)
            self.axis = axis
            self.split_mode = split_mode
            self.recombine = recombine
            self.units_total = units_total
            self.parts_specs = parts
            self.parts_layers = None

            if self.split_mode != "units":
                raise ValueError("SplitDense: split_mode soportado: 'units'.")
            if self.recombine != "concat":
                raise ValueError("SplitDense: recombine soportado: 'concat'.")

        def build(self, input_shape):
            self.parts_layers = []
            for spec in self.parts_specs:
                layer = tf.keras.layers.deserialize(spec)
                self.parts_layers.append(layer)
            super().build(input_shape)

        def call(self, inputs, training=None):
            outs = [layer(inputs, training=training) for layer in self.parts_layers]
            return tf.concat(outs, axis=self.axis)

        def get_config(self):
            cfg = super().get_config()
            cfg.update({
                "parts": self.parts_specs,
                "axis": self.axis,
                "split_mode": self.split_mode,
                "recombine": self.recombine,
                "units_total": self.units_total,
                "name": self.name
            })
            return cfg

        @classmethod
        def from_config(cls, config):
            return cls(**config)

    # ✅ AQUÍ ESTÁ EL CAMBIO CLAVE: custom_objects + compile=False
    model = tf.keras.models.load_model(
        LOCAL_UNTRAINED_MODEL_PATH,
        custom_objects={"SplitDense": SplitDense},
        compile=False
    )

    # Compilas tú (como ya hacías)
    model.compile(optimizer="sgd", loss="mean_squared_error", metrics=["mae", "mse", "mape"])
    tiempos["compilar_modelo_s"] = round(time.perf_counter() - t, 3)

    # Entrenamiento (epochs=epoch_final, initial_epoch=epoch_inicio)
    t = time.perf_counter()
    history = model.fit(X, Y, epochs=epoch_final, initial_epoch=epoch_inicio, verbose=0)
    tiempos["entrenamiento_s"] = round(time.perf_counter() - t, 3)

    # Guardar modelo
    t = time.perf_counter()
    model.save(LOCAL_TRAINED_MODEL_PATH)
    tiempos["guardar_modelo_s"] = round(time.perf_counter() - t, 3)

    # Subir modelo
    t = time.perf_counter()
    client = bs.get_blob_client(CONTAINER_NAME, TRAINED_MODEL_BLOB)
    with open(LOCAL_TRAINED_MODEL_PATH, "rb") as f:
        client.upload_blob(f, overwrite=True)
    tiempos["subir_modelo_s"] = round(time.perf_counter() - t, 3)

    # Recursos finales
    cpu_fin = proc.cpu_percent(interval=None)
    mem_fin = proc.memory_info().rss / (1024 ** 2)
    disk_fin = uso_disco_temporal()
    total_time = round(time.perf_counter() - start_total, 3)

    tiempos.update(
        {
            "modelo_origen": modelo_origen,
            "cpu_inicial_percent": cpu_ini,
            "cpu_final_percent": cpu_fin,
            "memoria_inicial_mb": round(mem_ini, 2),
            "memoria_final_mb": round(mem_fin, 2),
            "memoria_consumida_mb": round(mem_fin - mem_ini, 2),
            "disk_inicial_mb": round(disk_ini, 2),
            "disk_final_mb": round(disk_fin, 2),
            "disk_consumido_mb": round(disk_fin - disk_ini, 2),
            "total_s": total_time,
        }
    )

    resp = {
        "bloque_completado": [epoch_inicio, epoch_final],
        "epocas_totales": total_epochs,
        "tiempos": tiempos,
        "metricas_finales": {k: float(history.history[k][-1]) for k in history.history},
        "registros_totales": int(X.shape[0]),
    }

    return func.HttpResponse(
        json.dumps(resp, ensure_ascii=False, indent=2),
        status_code=200,
        mimetype="application/json",
    )
