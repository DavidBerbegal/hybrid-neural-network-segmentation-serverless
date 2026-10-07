import azure.functions as func
import json
import os
import tempfile
import time
import logging
import traceback

app = func.FunctionApp()

# Blob config por variables de entorno (recomendado)
STORAGE_CONNECTION_STRING = os.getenv(
    "STORAGE_CONNECTION_STRING"
)

CONTAINER_NAME = os.getenv("MODEL_CONTAINER_NAME", "modelos")
TRAINED_MODEL_BLOB = os.getenv("TRAINED_MODEL_BLOB", "modelo_entrenado.h5")

# Ruta local temporal
temp_dir = tempfile.gettempdir()
LOCAL_TRAINED_MODEL_PATH = os.path.join(temp_dir, TRAINED_MODEL_BLOB)


def _limpiar_config(layer_spec: dict) -> dict:
    """
    Mantiene coherencia con tu pipeline: si vienen configs con batch_input_shape, lo quitamos.
    (Esto solo afecta a la deserialización de specs internos si algún spec lo incluyese.)
    """
    if isinstance(layer_spec, dict) and "config" in layer_spec and isinstance(layer_spec["config"], dict):
        layer_spec["config"].pop("batch_input_shape", None)
    return layer_spec


@app.route(route="http_trigger_prediccion", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
def http_trigger_prediccion(req: func.HttpRequest) -> func.HttpResponse:
    # Imports pesados DENTRO (evita romper indexing del host)
    import numpy as np
    import psutil
    import tensorflow as tf
    from azure.storage.blob import BlobServiceClient

    try:
        tiempos = {}
        t_total_inicio = time.perf_counter()
        process = psutil.Process(os.getpid())

        # --- Monitorización inicial ---
        memoria_inicial = process.memory_info().rss / (1024**2)
        cpu_inicial = process.cpu_percent(interval=None)

        # --- Validación config ---
        if not STORAGE_CONNECTION_STRING:
            return func.HttpResponse(
                json.dumps({"error": "Falta STORAGE_CONNECTION_STRING en App Settings."}, ensure_ascii=False, indent=2),
                status_code=500,
                mimetype="application/json"
            )

        # --- Custom layer (por si el modelo entrenado incluye SplitDense) ---
        @tf.keras.utils.register_keras_serializable(package="custom")
        class SplitDense(tf.keras.layers.Layer):
            """
            Capa compuesta: aplica varias Dense en paralelo y concatena.
            Necesaria si el modelo entrenado contiene SplitDense.
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
                    spec = _limpiar_config(spec)
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
                })
                return cfg

            @classmethod
            def from_config(cls, config):
                return cls(**config)

        # --- Paso 1: Descargar modelo ---
        t1 = time.perf_counter()
        try:
            # limpiar modelo local previo (evita residuos)
            try:
                if os.path.exists(LOCAL_TRAINED_MODEL_PATH):
                    os.remove(LOCAL_TRAINED_MODEL_PATH)
            except Exception:
                pass

            blob_service = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
            blob_client = blob_service.get_blob_client(container=CONTAINER_NAME, blob=TRAINED_MODEL_BLOB)

            with open(LOCAL_TRAINED_MODEL_PATH, "wb") as f:
                f.write(blob_client.download_blob().readall())

            tiempos["descargar_modelo_s"] = round(time.perf_counter() - t1, 3)
        except Exception as e:
            tiempos["descargar_modelo_s"] = round(time.perf_counter() - t1, 3)
            return func.HttpResponse(
                json.dumps({"error": f"Error al descargar el modelo: {str(e)}", "tiempos": tiempos}, ensure_ascii=False, indent=2),
                status_code=500,
                mimetype="application/json"
            )

        # --- Paso 2: Cargar modelo ---
        t2 = time.perf_counter()
        try:
            # ✅ Clave: custom_objects + compile=False (inferencia estable)
            modelo = tf.keras.models.load_model(
                LOCAL_TRAINED_MODEL_PATH,
                custom_objects={"SplitDense": SplitDense},
                compile=False
            )
            tiempos["cargar_modelo_s"] = round(time.perf_counter() - t2, 3)
        except Exception as e:
            tiempos["cargar_modelo_s"] = round(time.perf_counter() - t2, 3)
            return func.HttpResponse(
                json.dumps({"error": f"Error al cargar el modelo: {str(e)}", "tiempos": tiempos}, ensure_ascii=False, indent=2),
                status_code=500,
                mimetype="application/json"
            )

        # --- Paso 3: Procesar entrada ---
        t3 = time.perf_counter()
        try:
            body = req.get_json()
            if "entrada" not in body:
                raise ValueError("Falta el campo 'entrada' en el body.")

            nueva_entrada = np.array(body["entrada"], dtype=np.float32)

            # Acepta [x1, x2] o [[x1, x2], [x1, x2], ...]
            if nueva_entrada.ndim == 1:
                nueva_entrada = nueva_entrada.reshape(1, -1)

            if nueva_entrada.ndim != 2:
                raise ValueError("El campo 'entrada' debe ser un vector 1D o una matriz 2D.")

            tiempos["procesar_entrada_s"] = round(time.perf_counter() - t3, 3)
        except Exception as e:
            tiempos["procesar_entrada_s"] = round(time.perf_counter() - t3, 3)
            return func.HttpResponse(
                json.dumps({"error": f"Error al procesar la entrada: {str(e)}", "tiempos": tiempos}, ensure_ascii=False, indent=2),
                status_code=400,
                mimetype="application/json"
            )

        # --- Paso 4: Predicción ---
        t4 = time.perf_counter()
        try:
            pred = modelo.predict(nueva_entrada, verbose=0)

            # soporta salida escalar típica [[y]]
            valor = float(np.array(pred).reshape(-1)[0])

            tiempos["predecir_s"] = round(time.perf_counter() - t4, 3)
        except Exception as e:
            tiempos["predecir_s"] = round(time.perf_counter() - t4, 3)
            return func.HttpResponse(
                json.dumps({"error": f"Error durante la predicción: {str(e)}", "tiempos": tiempos}, ensure_ascii=False, indent=2),
                status_code=500,
                mimetype="application/json"
            )

        # --- Monitorización final ---
        memoria_final = process.memory_info().rss / (1024**2)
        cpu_final = process.cpu_percent(interval=None)

        tiempos.update({
            "memoria_inicial_mb": round(memoria_inicial, 2),
            "memoria_final_mb": round(memoria_final, 2),
            "memoria_consumida_mb": round(memoria_final - memoria_inicial, 2),
            "cpu_inicial_percent": cpu_inicial,
            "cpu_final_percent": cpu_final,
            "total_s": round(time.perf_counter() - t_total_inicio, 3),
        })

        return func.HttpResponse(
            json.dumps({"prediccion": valor, "tiempos": tiempos}, ensure_ascii=False, indent=2),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        # Catch-all para que Azure no te devuelva “500 vacío”
        return func.HttpResponse(
            json.dumps({
                "error": str(e),
                "trace": traceback.format_exc()
            }, ensure_ascii=False, indent=2),
            status_code=500,
            mimetype="application/json"
        )
