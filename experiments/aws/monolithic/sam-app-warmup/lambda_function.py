import json, time, urllib.request, urllib.error, concurrent.futures

URLS = [
    "<TENSORFLOW_CONTAINER_URL>",  # contenedor
    "<PREDICTION_URL>",  # predicción
    "<TRAINING_URL>",  # entrenamiento
]

PER_URL_TIMEOUT = 2.0  # segundos

def _ping(url: str, timeout: float = PER_URL_TIMEOUT):
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(url, method="GET", headers={"User-Agent":"lambda-warmup/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            code = resp.getcode()
        return {"url": url, "status_code": code, "latencia_s": round(time.perf_counter()-t0, 3), "error": None}
    except urllib.error.HTTPError as e:
        return {"url": url, "status_code": e.code, "latencia_s": round(time.perf_counter()-t0, 3), "error": f"HTTPError: {e.reason}"}
    except Exception as e:
        return {"url": url, "status_code": None, "latencia_s": round(time.perf_counter()-t0, 3), "error": str(e)}

def lambda_handler(event, context):
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(URLS))) as ex:
        resultados = list(ex.map(_ping, URLS))
    return {
        "statusCode": 200,
        "headers": {"Content-Type":"application/json; charset=utf-8"},
        "body": json.dumps({
            "resultado":"warm-up completo",
            "detalles": resultados,
            "total_s": round(time.perf_counter()-t0, 3)
        }, ensure_ascii=False, indent=2)
    }
