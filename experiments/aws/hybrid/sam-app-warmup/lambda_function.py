import json
import logging
import os
import time

import psutil
import requests


logging.basicConfig(level=logging.INFO)


def lambda_handler(event, context):
    process = psutil.Process()
    tiempos = {}
    t_total_inicio = time.perf_counter()

    memoria_ini = process.memory_info().rss / (1024**2)
    cpu_ini = process.cpu_percent(interval=None)

    tiempos["memoria_inicial_mb"] = round(memoria_ini, 2)
    tiempos["cpu_inicial_percent"] = cpu_ini

    nombres_variables = [
        "URL_DATOS",
        "URL_MODELO",
        "URL_CAPA_ENTRADA",
        "URL_CAPA_OCULTA",
        "URL_CAPA_SALIDA",
        "URL_ENTRENAMIENTO",
        "URL_PREDICCION",
    ]

    urls = []

    for nombre in nombres_variables:
        url = os.environ.get(nombre, "")

        if not url:
            return {
                "statusCode": 500,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps(
                    {"error": f"Falta variable de entorno {nombre}."},
                    ensure_ascii=False,
                ),
            }

        urls.append(url)

    resultados = []

    for url in urls:
        t1 = time.perf_counter()

        try:
            response = requests.get(url, timeout=10)
            latencia = round(time.perf_counter() - t1, 3)

            resultados.append({
                "url": url,
                "status_code": response.status_code,
                "latencia_s": latencia,
                "error": None,
            })

            logging.info(
                "WARM-UP %s -> %s en %ss",
                url,
                response.status_code,
                latencia,
            )

        except Exception as error:
            latencia = round(time.perf_counter() - t1, 3)

            resultados.append({
                "url": url,
                "status_code": None,
                "latencia_s": latencia,
                "error": str(error),
            })

            logging.error(
                "WARM-UP %s error tras %ss: %s",
                url,
                latencia,
                error,
            )

    memoria_fin = process.memory_info().rss / (1024**2)
    cpu_fin = process.cpu_percent(interval=None)

    tiempos["memoria_final_mb"] = round(memoria_fin, 2)
    tiempos["cpu_final_percent"] = cpu_fin
    tiempos["total_s"] = round(
        time.perf_counter() - t_total_inicio, 3
    )

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(
            {
                "resultado": "warm-up completo",
                "detalles": resultados,
                "tiempos": tiempos,
            },
            ensure_ascii=False,
            indent=2,
        ),
    }