import azure.functions as func
import json
import logging
import time
import os
import tempfile

app = func.FunctionApp()

STORAGE_CONNECTION_STRING = os.getenv(
    "STORAGE_CONNECTION_STRING"
)

CONTAINER_NAME = os.getenv("MODEL_CONTAINER_NAME", "modelos")

def limpiar_config(layer_spec: dict) -> dict:
    if "config" in layer_spec and isinstance(layer_spec["config"], dict):
        layer_spec["config"].pop("batch_input_shape", None)
    return layer_spec

@app.route(route="http_trigger_modelo", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
def http_trigger_modelo(req: func.HttpRequest) -> func.HttpResponse:
    import psutil
    import requests
    from azure.storage.blob import BlobServiceClient
    import tensorflow as tf

    class SplitDense(tf.keras.layers.Layer):
        """
        Capa compuesta: aplica varias Dense en paralelo y concatena sus salidas.
        """
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

    def deserialize_layer(spec: dict) -> tf.keras.layers.Layer:
        spec = limpiar_config(spec)
        return tf.keras.layers.deserialize(spec, custom_objects={"SplitDense": SplitDense})

    # --- Monitoreo inicial ---
    t0 = time.perf_counter()
    proc = psutil.Process()
    cpu_ini = proc.cpu_percent(interval=None)
    mem_ini = proc.memory_info().rss / (1024**2)

    # --- Rutas temp ---
    tmp = tempfile.gettempdir()
    model_name = "modelo_sin_entrenar.h5"
    model_path = os.path.join(tmp, model_name)
    if os.path.exists(model_path):
        os.remove(model_path)

    tiempos = {}
    logging.info("Creando el modelo de red neuronal.")

    # --- Parse request ---
    try:
        body = req.get_json()
    except Exception:
        body = {}

    try:
        num_capas_ocultas = int(body.get("num_capas_ocultas", 1))
    except Exception:
        num_capas_ocultas = 1

    try:
        split_parts = int(body.get("split_parts", 1))
    except Exception:
        split_parts = 1

    split_layers = body.get("split_layers", None)
    tiempos["parse_request_s"] = round(time.perf_counter() - t0, 3)

    # Serverless layer endpoints
    url_ent = "<INPUT_LAYER_FUNCTION_URL>"
    url_occ = "<HIDDEN_LAYER_FUNCTION_URL>"
    url_sal = "<OUTPUT_LAYER_FUNCTION_URL>"

    capas = []

    # --- Capa entrada ---
    t1 = time.perf_counter()
    params_entrada = {"units": 5, "activation": "relu", "name": "entrada", "input_shape": [2]}
    r = requests.post(url_ent, json=params_entrada, timeout=30)
    tiempos["capa_entrada_s"] = round(time.perf_counter() - t1, 3)
    if r.status_code != 200:
        return func.HttpResponse(json.dumps({"error": r.text, "tiempos": tiempos}, ensure_ascii=False),
                                 status_code=500, mimetype="application/json")

    capas.append(deserialize_layer(r.json()))

    # --- Capas ocultas ---
    t2 = time.perf_counter()
    for i in range(num_capas_ocultas):
        t2i = time.perf_counter()
        idx_capa = i + 1

        partir_esta = False
        if split_parts and split_parts > 1:
            if split_layers is None:
                partir_esta = True
            else:
                try:
                    partir_esta = int(idx_capa) in [int(x) for x in split_layers]
                except Exception:
                    partir_esta = False

        params = {"units": 4, "activation": "relu", "name": f"oculta_{idx_capa}"}
        if partir_esta:
            params.update({"split_parts": split_parts, "split_mode": "units", "recombine": "concat"})

        r = requests.post(url_occ, json=params, timeout=30)
        if r.status_code != 200:
            return func.HttpResponse(json.dumps({"error": r.text, "tiempos": tiempos}, ensure_ascii=False),
                                     status_code=500, mimetype="application/json")

        capas.append(deserialize_layer(r.json()))
        tiempos[f"capa_oculta_{idx_capa}_s"] = round(time.perf_counter() - t2i, 3)

    tiempos["capas_ocultas_s"] = round(time.perf_counter() - t2, 3)

    # --- Capa salida ---
    t3 = time.perf_counter()
    params_sal = {"units": 1, "activation": "linear", "name": "salida"}
    r = requests.post(url_sal, json=params_sal, timeout=30)
    tiempos["capa_salida_s"] = round(time.perf_counter() - t3, 3)
    if r.status_code != 200:
        return func.HttpResponse(json.dumps({"error": r.text, "tiempos": tiempos}, ensure_ascii=False),
                                 status_code=500, mimetype="application/json")

    capas.append(deserialize_layer(r.json()))

    # --- Compilar ---
    t4 = time.perf_counter()
    modelo = tf.keras.Sequential(capas)
    modelo.compile(optimizer="sgd", loss="mean_squared_error", metrics=["mae", "mse", "mape"])
    tiempos["compilar_modelo_s"] = round(time.perf_counter() - t4, 3)

    # --- Guardar ---
    t5 = time.perf_counter()
    modelo.save(model_path)
    tiempos["guardar_modelo_s"] = round(time.perf_counter() - t5, 3)

    try:
        disk_fin = os.path.getsize(model_path) / (1024**2)
    except OSError:
        disk_fin = 0.0

    # --- Subir a Blob ---
    t6 = time.perf_counter()
    if not STORAGE_CONNECTION_STRING:
        return func.HttpResponse(
            json.dumps({"error": "Falta STORAGE_CONNECTION_STRING en App Settings.", "tiempos": tiempos}, ensure_ascii=False),
            status_code=500, mimetype="application/json"
        )

    blob_service = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
    blob_client = blob_service.get_blob_client(container=CONTAINER_NAME, blob=model_name)
    with open(model_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)

    tiempos["subir_blob_s"] = round(time.perf_counter() - t6, 3)

    # --- Monitoreo final ---
    t_total = time.perf_counter()
    cpu_fin = proc.cpu_percent(interval=None)
    mem_fin = proc.memory_info().rss / (1024**2)
    tiempos["total_s"] = round(t_total - t0, 3)

    resp = {
        "mensaje": "Modelo creado y almacenado.",
        "tiempos": tiempos,
        "particionado": {"split_parts": split_parts, "split_layers": split_layers},
        "recursos": {
            "cpu_inicial_percent": cpu_ini,
            "cpu_final_percent": cpu_fin,
            "memoria_inicial_mb": round(mem_ini, 2),
            "memoria_final_mb": round(mem_fin, 2),
            "memoria_consumida_mb": round(mem_fin - mem_ini, 2),
            "disk_final_mb": round(disk_fin, 2),
            "duration_s": round(t_total - t0, 3)
        }
    }
    return func.HttpResponse(json.dumps(resp, ensure_ascii=False, indent=2),
                             status_code=200, mimetype="application/json")
