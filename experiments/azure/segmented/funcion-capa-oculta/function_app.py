import azure.functions as func
import json
from keras.layers import Dense

app = func.FunctionApp()

@app.route(route="http_trigger_capa_oculta", auth_level=func.AuthLevel.ANONYMOUS)
def http_trigger_capa_oculta(req: func.HttpRequest) -> func.HttpResponse:
    try:
        # Parse request body
        req_body = req.get_json()
        units = req_body.get("units")
        activation = req_body.get("activation")
        name = req_body.get("name")

        input_shape = req_body.get("input_shape", None)
        
        # Construir la capa
        layer = Dense(
                units=units,
                activation=activation,
                name=name
        )
        
        # Serializar la capa
        result = {
            "module": "keras.layers",
            "class_name": "Dense",
            "config": layer.get_config()
        }
        return func.HttpResponse(
            json.dumps(result, ensure_ascii=False, indent=2),
            status_code=200,
            mimetype="application/json"
        )
    except Exception as e:
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=400,
            mimetype="application/json"
        )
