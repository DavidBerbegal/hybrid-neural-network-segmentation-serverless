from flask import Flask, request, jsonify
import base64
import logging
import awsgi
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import losses
import numpy as np
import tempfile
import time

# Forzar modo eager para evitar error .numpy()
tf.config.run_functions_eagerly(True)

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

def limpiar_config(layer_config):
    # Elimina batch_input_shape si existe (Dense no lo admite en deserialización directa)
    if isinstance(layer_config, dict):
        if 'config' in layer_config and isinstance(layer_config['config'], dict):
            layer_config['config'].pop('batch_input_shape', None)
    return layer_config

@app.route("/configure-layer", methods=["POST"])
def configure_model():
    app.logger.info(f"Headers: {dict(request.headers)}")
    app.logger.info(f"Raw data: {request.data.decode('utf-8')}")

    try:
        data = request.get_json()
        logging.info(f"Petición recibida en /configure-layer con data: {data}")

        if data is None:
            logging.warning("No se pudo decodificar el JSON")
            return jsonify({"error": "JSON inválido"}), 400

        layers = data.get("layers", [])
        if not layers:
            logging.warning("No se recibieron capas")
            return jsonify({"error": "No se recibieron capas"}), 400

        model = keras.Sequential()
        for layer in layers:
            if isinstance(layer, dict) and layer.get("class_name") and layer.get("config"):
                try:
                    model.add(keras.layers.deserialize(limpiar_config(layer)))
                except Exception as e:
                    logging.exception("No se pudo deserializar la capa")
                    return jsonify({"error": f"Error deserializando capa: {str(e)}"}), 400
            else:
                model.add(keras.layers.Dense(
                    units=layer.get("units", 1),
                    activation=layer.get("activation", "relu"),
                    name=layer.get("name")
                ))

        model.compile(optimizer='adam', loss=losses.MeanSquaredError())

        with tempfile.NamedTemporaryFile(suffix=".h5") as tmpfile:
            model.save(tmpfile.name)
            tmpfile.seek(0)
            model_bytes = tmpfile.read()

        return jsonify({
            "message": "Modelo real generado y serializado",
            "model_base64": base64.b64encode(model_bytes).decode("utf-8")
        })

    except Exception as e:
        logging.exception("Error inesperado en el contenedor")
        return jsonify({"error": str(e)}), 500

@app.route("/train-model", methods=["POST"])
def train_model():
    try:
        data = request.get_json()
        logging.info(f"Petición recibida en /train-model con data: {list(data.keys()) if isinstance(data, dict) else data}")

        model_b64 = data.get("model_base64")
        X = data.get("X")
        Y = data.get("Y")

        # Hiperparámetros opcionales (paridad con Azure)
        epochs = int(data.get("epochs", 1000))
        initial_epoch = int(data.get("initial_epoch", 0))
        opt_name = data.get("optimizer", "sgd")
        loss_name = data.get("loss", "mean_squared_error")
        metrics = data.get("metrics", ["mae", "mse", "mape"])

        if not isinstance(model_b64, str) or X is None or Y is None:
            logging.warning("Faltan model_base64, X o Y")
            return jsonify({"error": "Faltan model_base64, X o Y"}), 400

        try:
            model_bytes = base64.b64decode(model_b64)
        except Exception:
            logging.warning("model_base64 no es base64 válido")
            return jsonify({"error": "model_base64 no es base64 válido"}), 400

        # Cargar modelo base (.h5)
        with tempfile.NamedTemporaryFile(suffix=".h5") as tmpfile:
            tmpfile.write(model_bytes)
            tmpfile.flush()
            model = keras.models.load_model(
                tmpfile.name, custom_objects={'mse': losses.MeanSquaredError()}
            )

        # (Re)compilar con lo recibido
        optimizer = keras.optimizers.get(opt_name)
        loss_obj = losses.get(loss_name) if isinstance(loss_name, str) else loss_name
        model.compile(optimizer=optimizer, loss=loss_obj, metrics=metrics)

        # Datos
        X_np = np.array(X, dtype=float)
        Y_np = np.array(Y, dtype=float)

        # Entrenamiento + cronómetro
        t0 = time.perf_counter()
        history = model.fit(
            X_np, Y_np, epochs=epochs, initial_epoch=initial_epoch, verbose=0
        )
        train_time_s = round(time.perf_counter() - t0, 3)

        # Serializar modelo entrenado
        with tempfile.NamedTemporaryFile(suffix=".h5") as tmpfile_out:
            model.save(tmpfile_out.name)
            tmpfile_out.flush()
            tmpfile_out.seek(0)
            trained_model_bytes = tmpfile_out.read()
        trained_model_b64 = base64.b64encode(trained_model_bytes).decode("utf-8")

        # EXTRAER MÉTRICAS FINALES (formato que espera tu orquestador)
        metricas_finales = {}
        for key in ["loss", "mae", "mse", "mape"]:
            if key in history.history and len(history.history[key]) > 0:
                metricas_finales[key] = float(history.history[key][-1])

        return jsonify({
            "message": "Modelo entrenado y serializado",
            "model_base64": trained_model_b64,
            "metrics": metricas_finales,   # <— IMPORTANTÍSIMO para tu CSV
            "train_time_s": train_time_s
        }), 200

    except Exception as e:
        logging.exception("Error inesperado en /train-model")
        return jsonify({"error": str(e)}), 500

@app.route("/capa-entrada", methods=["POST"])
def capa_entrada():
    try:
        data = request.get_json()
        logging.info(f"Petición recibida en /capa-entrada con data: {data}")

        units = data.get("units")
        input_shape = data.get("input_shape")
        activation = data.get("activation")
        name = data.get("name", "capa_entrada_dense")

        if units is None or input_shape is None or activation is None:
            return jsonify({
                "error": "Faltan parámetros obligatorios: 'units', 'input_shape', 'activation'."
            }), 400

        # Crear la capa Dense de entrada
        layer = keras.layers.Dense(
            units=units,
            activation=activation,
            input_shape=tuple(input_shape),
            name=name
        )

        # Ajustar config para asegurar batch_input_shape correcto
        layer_config = layer.get_config()
        layer_config['batch_input_shape'] = (None, *input_shape)

        result = {
            "module": "keras.layers",
            "class_name": "Dense",
            "config": layer_config
        }

        return jsonify(result), 200

    except Exception as e:
        logging.exception("Error creando capa de entrada")
        return jsonify({"error": str(e)}), 500

@app.route("/capa-oculta", methods=["POST"])
def capa_oculta():
    try:
        data = request.get_json()
        logging.info(f"Petición recibida en /capa-oculta con data: {data}")

        units = data.get("units")
        activation = data.get("activation")
        name = data.get("name")

        # Igual que Azure: no usamos input_shape aquí.
        layer = keras.layers.Dense(
            units=units,
            activation=activation,
            name=name
        )

        result = {
            "module": "keras.layers",
            "class_name": "Dense",
            "config": layer.get_config()
        }
        return jsonify(result), 200

    except Exception as e:
        logging.exception("Error creando capa oculta")
        return jsonify({"error": str(e)}), 400

@app.route("/capa-salida", methods=["POST"])
def capa_salida():
    try:
        data = request.get_json()
        logging.info(f"Petición recibida en /capa-salida con data: {data}")

        units = data.get("units", 1)
        activation = data.get("activation", "linear")
        name = data.get("name", "capa_salida_dense")

        layer = keras.layers.Dense(
            units=units,
            activation=activation,
            name=name
        )

        result = {
            "module": "keras.layers",
            "class_name": "Dense",
            "config": layer.get_config()
        }
        return jsonify(result), 200

    except Exception as e:
        logging.exception("Error creando capa de salida")
        return jsonify({"error": str(e)}), 400

@app.route("/predict-model", methods=["POST"])
def predict_model():
    try:
        data = request.get_json()
        logging.info(f"Petición recibida en /predict-model con keys: {list(data.keys()) if isinstance(data, dict) else data}")

        model_b64 = data.get("model_base64")
        X = data.get("X")

        if not isinstance(model_b64, str) or X is None:
            return jsonify({"error": "Faltan model_base64 o X"}), 400

        try:
            model_bytes = base64.b64decode(model_b64)
        except Exception:
            return jsonify({"error": "model_base64 no es base64 válido"}), 400

        with tempfile.NamedTemporaryFile(suffix=".h5") as tmpfile:
            tmpfile.write(model_bytes)
            tmpfile.flush()
            model = keras.models.load_model(tmpfile.name, custom_objects={'mse': losses.MeanSquaredError()})

        X_np = np.array(X, dtype=float)
        # Garantizar 2D
        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)

        preds = model.predict(X_np, verbose=0)
        return jsonify({
            "predictions": preds.tolist()
        }), 200

    except Exception as e:
        logging.exception("Error inesperado en /predict-model")
        return jsonify({"error": str(e)}), 500

def handler(event, context):
    # Detectar evento Lambda URL (no tiene 'httpMethod')
    if "httpMethod" not in event:
        method = event.get("requestContext", {}).get("http", {}).get("method", "POST")
        path = event.get("rawPath", "/")
        headers = event.get("headers", {})
        body = event.get("body", "")
        if event.get("isBase64Encoded", False):
            import base64
            body = base64.b64decode(body).decode("utf-8")
        query_params = event.get("queryStringParameters", {})

        api_event = {
            "httpMethod": method,
            "path": path,
            "headers": headers,
            "body": body,
            "isBase64Encoded": False,
            "queryStringParameters": query_params
        }
        return awsgi.response(app, api_event, context)
    else:
        return awsgi.response(app, event, context)