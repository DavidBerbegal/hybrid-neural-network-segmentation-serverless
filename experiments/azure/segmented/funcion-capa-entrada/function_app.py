import azure.functions as func
import logging
import json
from keras.layers import Dense

app = func.FunctionApp()

@app.route(route="http_trigger_capa_entrada", auth_level=func.AuthLevel.ANONYMOUS)
def http_trigger_capa_entrada(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "GET":
        return func.HttpResponse("Función capa_entrada activa", status_code=200)
    
    if req.method == "POST":
        try:
            req_body = req.get_json()
            units = req_body.get("units")
            input_shape = req_body.get("input_shape")
            activation = req_body.get("activation")
            name = req_body.get("name", "capa_entrada_dense")

            if units is None or input_shape is None or activation is None:
                return func.HttpResponse(
                    "Faltan parámetros obligatorios: 'units', 'input_shape', 'activation'.",
                    status_code=400
                )

            # Crear la capa Dense de entrada (usa input_shape)
            layer = Dense(
                units=units,
                activation=activation,
                input_shape=tuple(input_shape),
                name=name
            )

            # Obtener el config y asegurarse que batch_input_shape está bien
            layer_config = layer.get_config()
            layer_config['batch_input_shape'] = (None, *input_shape)  # Esto es lo CLAVE

            result = {
                "module": "keras.layers",
                "class_name": "Dense",
                "config": layer_config
            }

            return func.HttpResponse(
                json.dumps(result, ensure_ascii=False, indent=2),
                mimetype="application/json",
                status_code=200
            )
        except Exception as e:
            logging.error(f"Error creando capa de entrada: {str(e)}")
            return func.HttpResponse(f"Error: {str(e)}", status_code=500)

    return func.HttpResponse("Método no soportado", status_code=405)
