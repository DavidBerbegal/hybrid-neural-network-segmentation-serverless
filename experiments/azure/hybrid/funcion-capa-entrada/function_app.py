import os
import sys
import json
import logging

# ------------------------------------------------------------
# Garantizar que se cargan dependencias del proyecto en Azure
# ------------------------------------------------------------
SITE = os.path.join(os.path.dirname(__file__), ".python_packages", "lib", "site-packages")
if SITE not in sys.path:
    sys.path.insert(0, SITE)

# Si el worker ya cargó protobuf, lo expulsamos para que use el nuestro (protobuf 5.x)
for m in list(sys.modules.keys()):
    if m == "google" or m.startswith("google.protobuf"):
        del sys.modules[m]

import azure.functions as func

app = func.FunctionApp()

@app.route(
    route="http_trigger_capa_entrada",
    auth_level=func.AuthLevel.ANONYMOUS,
    methods=["GET", "POST"],
)
def http_trigger_capa_entrada(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "GET":
        return func.HttpResponse("Función capa_entrada activa", status_code=200)

    if req.method == "POST":
        try:
            from tensorflow import keras
            Dense = keras.layers.Dense

            req_body = req.get_json()
            units = req_body.get("units")
            input_shape = req_body.get("input_shape")
            activation = req_body.get("activation")
            name = req_body.get("name", "capa_entrada_dense")

            if units is None or input_shape is None or activation is None:
                return func.HttpResponse(
                    "Faltan parámetros obligatorios: 'units', 'input_shape', 'activation'.",
                    status_code=400,
                )

            layer = Dense(
                units=units,
                activation=activation,
                input_shape=tuple(input_shape),
                name=name,
            )

            layer_config = layer.get_config()
            layer_config["batch_input_shape"] = (None, *input_shape)

            result = {
                "module": "keras.layers",
                "class_name": "Dense",
                "config": layer_config,
            }

            return func.HttpResponse(
                json.dumps(result, ensure_ascii=False, indent=2),
                mimetype="application/json",
                status_code=200,
            )

        except Exception as e:
            logging.exception("Error creando capa de entrada")
            return func.HttpResponse(f"Error: {str(e)}", status_code=500)

    return func.HttpResponse("Método no soportado", status_code=405)
