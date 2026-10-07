import azure.functions as func
import requests
import logging
import time
import psutil
import json

app = func.FunctionApp()

@app.route(route="http_trigger_warm_up_monolitica", auth_level=func.AuthLevel.ANONYMOUS)
def http_trigger_warm_up_monolitica(req: func.HttpRequest) -> func.HttpResponse:
    process = psutil.Process()
    tiempos = {}
    t_total = time.perf_counter()

    # Monitor inicial
    memoria_ini = process.memory_info().rss / 1024**2
    cpu_ini = process.cpu_percent(interval=None)
    tiempos["memoria_inicial_mb"] = round(memoria_ini, 2)
    tiempos["cpu_inicial_percent"] = cpu_ini

    urls = [
        "<TRAINING_FUNCTION_URL>",
        "<PREDICTION_FUNCTION_URL>",
    ]
    resultados = []

    for url in urls:
        t1 = time.perf_counter()
        try:
            resp = requests.get(url, timeout=10)
            elapsed = round(time.perf_counter() - t1, 3)
            resultados.append({
                "url": url,
                "status_code": resp.status_code,
                "latencia_s": elapsed,
                "error": None
            })
            logging.info(f"WARM-UP ✅ {url} → {resp.status_code} en {elapsed}s")
        except Exception as e:
            elapsed = round(time.perf_counter() - t1, 3)
            resultados.append({
                "url": url,
                "status_code": None,
                "latencia_s": elapsed,
                "error": str(e)
            })
            logging.error(f"WARM-UP ❌ {url} error tras {elapsed}s: {e}")

    # Monitor final
    memoria_fin = process.memory_info().rss / 1024**2
    cpu_fin = process.cpu_percent(interval=None)
    tiempos["memoria_final_mb"] = round(memoria_fin, 2)
    tiempos["cpu_final_percent"] = cpu_fin
    tiempos["total_s"] = round(time.perf_counter() - t_total, 3)

    payload = {
        "resultado": "warm-up completo",
        "detalles": resultados,
        "tiempos": tiempos
    }

    return func.HttpResponse(
        json.dumps(payload, ensure_ascii=False, indent=2),
        status_code=200,
        mimetype="application/json"
    )
