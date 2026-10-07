import azure.functions as func
import json

app = func.FunctionApp()

@app.route(route="http_trigger_capa_salida", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
def http_trigger_capa_salida(req: func.HttpRequest) -> func.HttpResponse:
    try:
        from tensorflow import keras

        req_body = req.get_json()
        units = req_body.get("units", 1)
        activation = req_body.get("activation", "linear")
        name = req_body.get("name", "capa_salida_dense")

        layer = keras.layers.Dense(
            units=units,
            activation=activation,
            name=name
        )

        layer_config = {
            "module": "keras.layers",
            "class_name": "Dense",
            "config": layer.get_config()
        }

        return func.HttpResponse(
            json.dumps(layer_config, ensure_ascii=False, indent=4),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        return func.HttpResponse(
            f"Error: {str(e)}",
            status_code=400
        )
