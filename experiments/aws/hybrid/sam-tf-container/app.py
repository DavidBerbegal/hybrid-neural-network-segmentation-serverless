from flask import Flask, request, jsonify

import base64
import logging
import tempfile
import time

import awsgi
import numpy as np
import tensorflow as tf
from tensorflow import keras


app = Flask(__name__)
logging.basicConfig(level=logging.INFO)


class SplitDense(tf.keras.layers.Layer):
    """
    Misma lógica que Azure híbrido:
    aplica cada subcapa a la misma entrada y concatena sus salidas.
    """

    def __init__(
        self,
        parts,
        axis=-1,
        split_mode="units",
        recombine="concat",
        units_total=None,
        name=None,
        **kwargs,
    ):
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
        outs = [
            layer(inputs, training=training)
            for layer in self.parts_layers
        ]
        return tf.concat(outs, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({
            "parts": self.parts_specs,
            "axis": self.axis,
            "split_mode": self.split_mode,
            "recombine": self.recombine,
            "units_total": self.units_total,
            "name": self.name,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def limpiar_config(layer_config):
    if isinstance(layer_config, dict):
        config = layer_config.get("config")

        if isinstance(config, dict):
            config.pop("batch_input_shape", None)

    return layer_config


def distribuir_units(units, split_parts):
    base = units // split_parts
    resto = units % split_parts

    return [
        base + (1 if i < resto else 0)
        for i in range(split_parts)
    ]


def dense_spec(layer):
    return {
        "module": "keras.layers",
        "class_name": "Dense",
        "config": layer.get_config(),
    }


def cargar_modelo(model_bytes):
    with tempfile.NamedTemporaryFile(suffix=".h5") as tmpfile:
        tmpfile.write(model_bytes)
        tmpfile.flush()

        return keras.models.load_model(
            tmpfile.name,
            custom_objects={"SplitDense": SplitDense},
            compile=False,
        )


def serializar_modelo(model):
    with tempfile.NamedTemporaryFile(suffix=".h5") as tmpfile:
        model.save(tmpfile.name)
        tmpfile.seek(0)
        model_bytes = tmpfile.read()

    return base64.b64encode(model_bytes).decode("utf-8")


@app.route("/capa-entrada", methods=["POST"])
def capa_entrada():
    try:
        data = request.get_json()

        units = data.get("units")
        input_shape = data.get("input_shape")
        activation = data.get("activation")
        name = data.get("name", "capa_entrada_dense")

        if units is None or input_shape is None or activation is None:
            return jsonify({
                "error": (
                    "Faltan parámetros obligatorios: "
                    "'units', 'input_shape', 'activation'."
                )
            }), 400

        layer = keras.layers.Dense(
            units=units,
            activation=activation,
            input_shape=tuple(input_shape),
            name=name,
        )

        config = layer.get_config()
        config["batch_input_shape"] = (None, *input_shape)

        return jsonify({
            "module": "keras.layers",
            "class_name": "Dense",
            "config": config,
        }), 200

    except Exception as error:
        logging.exception("Error creando capa de entrada")
        return jsonify({"error": str(error)}), 500


@app.route("/capa-oculta", methods=["POST"])
def capa_oculta():
    try:
        data = request.get_json()

        units = data.get("units")
        activation = data.get("activation")
        name = data.get("name")

        split_parts = data.get("split_parts", 1)
        split_mode = data.get("split_mode", "units")
        recombine = data.get("recombine", "concat")
        part_units = data.get("part_units")

        if units is None or activation is None or name is None:
            return jsonify({
                "error": (
                    "Faltan parámetros obligatorios: "
                    "'units', 'activation', 'name'."
                )
            }), 400

        try:
            units = int(units)
            split_parts = int(split_parts)
        except Exception:
            return jsonify({
                "error": "'units' y 'split_parts' deben ser enteros."
            }), 400

        if units <= 0:
            return jsonify({"error": "'units' debe ser > 0."}), 400

        if split_parts < 1:
            return jsonify({
                "error": "'split_parts' debe ser >= 1."
            }), 400

        if split_parts == 1:
            layer = keras.layers.Dense(
                units=units,
                activation=activation,
                name=name,
            )
            return jsonify(dense_spec(layer)), 200

        if split_mode != "units":
            return jsonify({
                "error": "Por ahora solo se soporta split_mode='units'."
            }), 400

        if recombine != "concat":
            return jsonify({
                "error": "Por ahora solo se soporta recombine='concat'."
            }), 400

        if part_units is not None:
            if (
                not isinstance(part_units, list)
                or len(part_units) != split_parts
            ):
                return jsonify({
                    "error": (
                        "'part_units' debe ser una lista "
                        "con longitud == split_parts."
                    )
                }), 400

            try:
                part_units = [int(value) for value in part_units]
            except Exception:
                return jsonify({
                    "error": "'part_units' debe contener enteros."
                }), 400

            if any(value <= 0 for value in part_units):
                return jsonify({
                    "error": "Todos los valores de 'part_units' deben ser > 0."
                }), 400

            if sum(part_units) != units:
                return jsonify({
                    "error": "La suma de 'part_units' debe ser igual a 'units'."
                }), 400
        else:
            part_units = distribuir_units(units, split_parts)

        parts = []

        for index, part_size in enumerate(part_units, start=1):
            layer = keras.layers.Dense(
                units=part_size,
                activation=activation,
                name=f"{name}__part{index}",
            )
            parts.append(dense_spec(layer))

        return jsonify({
            "module": "custom.layers",
            "class_name": "SplitDense",
            "config": {
                "name": name,
                "split_mode": "units",
                "recombine": "concat",
                "axis": -1,
                "units_total": units,
                "parts": parts,
            },
        }), 200

    except Exception as error:
        logging.exception("Error creando capa oculta")
        return jsonify({"error": str(error)}), 400


@app.route("/capa-salida", methods=["POST"])
def capa_salida():
    try:
        data = request.get_json()

        layer = keras.layers.Dense(
            units=data.get("units", 1),
            activation=data.get("activation", "linear"),
            name=data.get("name", "capa_salida_dense"),
        )

        return jsonify(dense_spec(layer)), 200

    except Exception as error:
        logging.exception("Error creando capa de salida")
        return jsonify({"error": str(error)}), 400


@app.route("/configure-layer", methods=["POST"])
def configure_model():
    try:
        data = request.get_json()

        if data is None:
            return jsonify({"error": "JSON inválido"}), 400

        layers = data.get("layers", [])

        if not layers:
            return jsonify({"error": "No se recibieron capas"}), 400

        tiempos = {}
        capas = []

        start = time.perf_counter()

        for spec in layers:
            try:
                layer = keras.layers.deserialize(
                    limpiar_config(spec),
                    custom_objects={"SplitDense": SplitDense},
                )
                capas.append(layer)
            except Exception as error:
                logging.exception("No se pudo deserializar la capa")
                return jsonify({
                    "error": f"Error deserializando capa: {error}"
                }), 400

        tiempos["deserializar_capas_s"] = round(
            time.perf_counter() - start, 3
        )

        start = time.perf_counter()

        model = keras.Sequential(capas)
        model.compile(
            optimizer="sgd",
            loss="mean_squared_error",
            metrics=["mae", "mse", "mape"],
        )

        tiempos["compilar_modelo_s"] = round(
            time.perf_counter() - start, 3
        )

        start = time.perf_counter()
        model_base64 = serializar_modelo(model)

        tiempos["serializar_modelo_s"] = round(
            time.perf_counter() - start, 3
        )

        return jsonify({
            "message": "Modelo real generado y serializado",
            "model_base64": model_base64,
            "tiempos": tiempos,
        }), 200

    except Exception as error:
        logging.exception("Error en /configure-layer")
        return jsonify({"error": str(error)}), 500


@app.route("/train-model", methods=["POST"])
def train_model():
    try:
        data = request.get_json()

        model_b64 = data.get("model_base64")
        X = data.get("X")
        Y = data.get("Y")

        if not isinstance(model_b64, str) or X is None or Y is None:
            return jsonify({
                "error": "Faltan model_base64, X o Y"
            }), 400

        epochs = int(data.get("epochs", 1000))
        initial_epoch = int(data.get("initial_epoch", 0))

        try:
            model_bytes = base64.b64decode(model_b64)
        except Exception:
            return jsonify({
                "error": "model_base64 no es base64 válido"
            }), 400

        tiempos = {}

        start = time.perf_counter()
        model = cargar_modelo(model_bytes)
        tiempos["cargar_modelo_s"] = round(
            time.perf_counter() - start, 3
        )

        start = time.perf_counter()

        model.compile(
            optimizer="sgd",
            loss="mean_squared_error",
            metrics=["mae", "mse", "mape"],
        )

        tiempos["compilar_modelo_s"] = round(
            time.perf_counter() - start, 3
        )

        X_np = np.array(X, dtype=np.float32)
        Y_np = np.array(Y, dtype=np.float32).reshape(-1, 1)

        start = time.perf_counter()

        history = model.fit(
            X_np,
            Y_np,
            epochs=epochs,
            initial_epoch=initial_epoch,
            verbose=0,
        )

        train_time_s = round(time.perf_counter() - start, 3)
        tiempos["entrenamiento_s"] = train_time_s

        start = time.perf_counter()
        trained_model_b64 = serializar_modelo(model)

        tiempos["serializar_modelo_s"] = round(
            time.perf_counter() - start, 3
        )

        metricas_finales = {}

        for key in ["loss", "mae", "mse", "mape"]:
            if key in history.history and history.history[key]:
                metricas_finales[key] = float(
                    history.history[key][-1]
                )

        return jsonify({
            "message": "Modelo entrenado y serializado",
            "model_base64": trained_model_b64,
            "metrics": metricas_finales,
            "train_time_s": train_time_s,
            "tiempos": tiempos,
        }), 200

    except Exception as error:
        logging.exception("Error en /train-model")
        return jsonify({"error": str(error)}), 500


@app.route("/predict-model", methods=["POST"])
def predict_model():
    try:
        data = request.get_json()

        model_b64 = data.get("model_base64")
        X = data.get("X")

        if not isinstance(model_b64, str) or X is None:
            return jsonify({
                "error": "Faltan model_base64 o X"
            }), 400

        try:
            model_bytes = base64.b64decode(model_b64)
        except Exception:
            return jsonify({
                "error": "model_base64 no es base64 válido"
            }), 400

        tiempos = {}

        start = time.perf_counter()
        model = cargar_modelo(model_bytes)

        tiempos["cargar_modelo_s"] = round(
            time.perf_counter() - start, 3
        )

        X_np = np.array(X, dtype=np.float32)

        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)

        start = time.perf_counter()
        predictions = model.predict(X_np, verbose=0)

        predict_time_s = round(time.perf_counter() - start, 3)
        tiempos["prediccion_s"] = predict_time_s

        return jsonify({
            "predictions": predictions.tolist(),
            "predict_time_s": predict_time_s,
            "tiempos": tiempos,
        }), 200

    except Exception as error:
        logging.exception("Error en /predict-model")
        return jsonify({"error": str(error)}), 500


def handler(event, context):
    if "httpMethod" not in event:
        method = (
            event.get("requestContext", {})
            .get("http", {})
            .get("method", "POST")
        )

        body = event.get("body") or ""

        if event.get("isBase64Encoded", False):
            body = base64.b64decode(body).decode("utf-8")

        api_event = {
            "httpMethod": method,
            "path": event.get("rawPath", "/"),
            "headers": event.get("headers") or {},
            "body": body,
            "isBase64Encoded": False,
            "queryStringParameters": event.get("queryStringParameters"),
        }

        return awsgi.response(app, api_event, context)

    return awsgi.response(app, event, context)