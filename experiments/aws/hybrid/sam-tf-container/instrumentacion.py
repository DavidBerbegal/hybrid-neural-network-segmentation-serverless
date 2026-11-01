import time

import psutil
from flask import g

from app import app, handler as original_handler


RUTAS_INSTRUMENTADAS = {
    "/configure-layer",
    "/train-model",
    "/predict-model",
}


@app.before_request
def iniciar_medicion():
    from flask import request

    if request.path not in RUTAS_INSTRUMENTADAS:
        return

    inicio = time.perf_counter()
    process = psutil.Process()
    cpu_inicial = process.cpu_percent(interval=None)
    memoria_inicial = process.memory_info().rss / (1024**2)

    g.medicion_contenedor = {
        "process": process,
        "inicio": inicio,
        "cpu_inicial": cpu_inicial,
        "memoria_inicial": memoria_inicial,
    }


@app.after_request
def finalizar_medicion(response):
    medicion = getattr(g, "medicion_contenedor", None)

    if medicion is None:
        return response

    fin = time.perf_counter()
    process = medicion["process"]
    cpu_final = process.cpu_percent(interval=None)
    memoria_final = process.memory_info().rss / (1024**2)

    payload = response.get_json(silent=True)

    if not isinstance(payload, dict):
        return response

    payload["recursos_contenedor"] = {
        "cpu_inicial_percent": medicion["cpu_inicial"],
        "cpu_final_percent": cpu_final,
        "memoria_inicial_mb": round(
            medicion["memoria_inicial"], 2
        ),
        "memoria_final_mb": round(memoria_final, 2),
        "memoria_consumida_mb": round(
            memoria_final - medicion["memoria_inicial"], 2
        ),
        "duration_s": round(fin - medicion["inicio"], 3),
    }

    response.set_data(app.json.dumps(payload))
    return response


def handler(event, context):
    return original_handler(event, context)