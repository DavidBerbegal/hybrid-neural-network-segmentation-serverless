import os
import json
import time
import base64
import logging

import requests
import boto3
import psutil


logging.basicConfig(level=logging.INFO)

URL_CAPA_ENTRADA = os.environ.get("URL_CAPA_ENTRADA", "")
URL_CAPA_OCULTA = os.environ.get("URL_CAPA_OCULTA", "")
URL_CAPA_SALIDA = os.environ.get("URL_CAPA_SALIDA", "")
URL_CONTENEDOR_CONFIG = os.environ.get("URL_CONTENEDOR_CONFIG", "")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "")
S3_KEY_H5 = os.environ.get("S3_KEY_H5", "modelo_sin_entrenar.h5")

s3 = boto3.client("s3")


def _resp(status, body):
    if isinstance(body, (dict, list)):
        body = json.dumps(body, ensure_ascii=False, indent=2)

    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": body,
    }


def _parse_method(event):
    return (
        event.get("httpMethod")
        or event.get("requestContext", {}).get("http", {}).get("method")
        or "GET"
    ).upper()


def lambda_handler(event, context):
    method = _parse_method(event)

    if method == "GET":
        return _resp(
            200,
            "FunciÃ³n modelo activa (AWS orquesta y el contenedor crea el .h5)",
        )

    if method != "POST":
        return _resp(405, "MÃ©todo no soportado")

    t0 = time.perf_counter()
    proc = psutil.Process()
    cpu_ini = proc.cpu_percent(interval=None)
    mem_ini = proc.memory_info().rss / (1024**2)
    tiempos = {}

    try:
        body_raw = event.get("body") or "{}"

        if event.get("isBase64Encoded", False):
            body_raw = base64.b64decode(body_raw).decode("utf-8")

        body = json.loads(body_raw)
    except Exception:
        body = {}

    try:
        num_capas_ocultas = int(body.get("num_capas_ocultas", 1))
    except Exception:
        num_capas_ocultas = 1

    try:
        split_parts = int(body.get("split_parts", 1))
    except Exception:
        split_parts = 1

    split_layers = body.get("split_layers", None)
    tiempos["parse_request_s"] = round(time.perf_counter() - t0, 3)

    for key, value in {
        "URL_CAPA_ENTRADA": URL_CAPA_ENTRADA,
        "URL_CAPA_OCULTA": URL_CAPA_OCULTA,
        "URL_CAPA_SALIDA": URL_CAPA_SALIDA,
        "URL_CONTENEDOR_CONFIG": URL_CONTENEDOR_CONFIG,
        "S3_BUCKET_NAME": S3_BUCKET_NAME,
    }.items():
        if not value:
            return _resp(
                500,
                {
                    "error": f"Falta variable de entorno {key}",
                    "tiempos": tiempos,
                },
            )

    ent = URL_CAPA_ENTRADA.rstrip("/")
    occ = URL_CAPA_OCULTA.rstrip("/")
    sal = URL_CAPA_SALIDA.rstrip("/")
    cfg = URL_CONTENEDOR_CONFIG.rstrip("/")

    try:
        t1 = time.perf_counter()

        params_entrada = {
            "units": 5,
            "activation": "relu",
            "name": "entrada",
            "input_shape": [2],
        }

        response = requests.post(ent, json=params_entrada, timeout=180)
        tiempos["capa_entrada_s"] = round(
            time.perf_counter() - t1, 3
        )

        if response.status_code != 200:
            return _resp(
                500,
                {"error": response.text, "tiempos": tiempos},
            )

        capas = [response.json()]

        t2 = time.perf_counter()

        for i in range(num_capas_ocultas):
            t2i = time.perf_counter()
            idx_capa = i + 1
            partir_esta = False

            if split_parts and split_parts > 1:
                if split_layers is None:
                    partir_esta = True
                else:
                    try:
                        partir_esta = int(idx_capa) in [
                            int(value) for value in split_layers
                        ]
                    except Exception:
                        partir_esta = False

            params = {
                "units": 4,
                "activation": "relu",
                "name": f"oculta_{idx_capa}",
            }

            if partir_esta:
                params.update({
                    "split_parts": split_parts,
                    "split_mode": "units",
                    "recombine": "concat",
                })

            response = requests.post(occ, json=params, timeout=180)

            if response.status_code != 200:
                return _resp(
                    500,
                    {"error": response.text, "tiempos": tiempos},
                )

            capas.append(response.json())
            tiempos[f"capa_oculta_{idx_capa}_s"] = round(
                time.perf_counter() - t2i, 3
            )

        tiempos["capas_ocultas_s"] = round(
            time.perf_counter() - t2, 3
        )

        t3 = time.perf_counter()

        params_salida = {
            "units": 1,
            "activation": "linear",
            "name": "salida",
        }

        response = requests.post(sal, json=params_salida, timeout=180)
        tiempos["capa_salida_s"] = round(
            time.perf_counter() - t3, 3
        )

        if response.status_code != 200:
            return _resp(
                500,
                {"error": response.text, "tiempos": tiempos},
            )

        capas.append(response.json())

        t4 = time.perf_counter()

        response = requests.post(
            cfg,
            json={"layers": capas},
            timeout=300,
        )

        tiempos["cont_config_s"] = round(
            time.perf_counter() - t4, 3
        )

        if response.status_code != 200:
            return _resp(
                500,
                {"error": response.text, "tiempos": tiempos},
            )

        data = response.json()
        model_b64 = data.get("model_base64")

        if not model_b64:
            return _resp(
                500,
                {
                    "error": "El contenedor no devolviÃ³ 'model_base64'",
                    "tiempos": tiempos,
                },
            )

        tiempos_contenedor = data.get("tiempos", {})

        if "compilar_modelo_s" in tiempos_contenedor:
            tiempos["compilar_modelo_s"] = (
                tiempos_contenedor["compilar_modelo_s"]
            )

        t_decode = time.perf_counter()
        model_bytes = base64.b64decode(model_b64)
        tiempos["decodificar_modelo_s"] = round(
            time.perf_counter() - t_decode, 3
        )

        t5 = time.perf_counter()

        s3.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=S3_KEY_H5,
            Body=model_bytes,
        )

        tiempos["subir_s3_s"] = round(
            time.perf_counter() - t5, 3
        )

        t_total = time.perf_counter()
        cpu_fin = proc.cpu_percent(interval=None)
        mem_fin = proc.memory_info().rss / (1024**2)
        tiempos["total_s"] = round(t_total - t0, 3)

        return _resp(
            200,
            {
                "mensaje": "Modelo creado y almacenado.",
                "tiempos": tiempos,
                "tiempos_contenedor": tiempos_contenedor,
                "recursos_contenedor": data.get("recursos_contenedor"),
                "particionado": {
                    "split_parts": split_parts,
                    "split_layers": split_layers,
                },
                "modelo_bytes": len(model_bytes),
                "recursos": {
                    "cpu_inicial_percent": cpu_ini,
                    "cpu_final_percent": cpu_fin,
                    "memoria_inicial_mb": round(mem_ini, 2),
                    "memoria_final_mb": round(mem_fin, 2),
                    "memoria_consumida_mb": round(
                        mem_fin - mem_ini, 2
                    ),
                    "disk_inicial_mb": 0.0,
                    "disk_final_mb": 0.0,
                    "disk_consumido_mb": 0.0,
                    "duration_s": round(t_total - t0, 3),
                },
            },
        )

    except Exception as error:
        logging.exception("Error creando o almacenando el modelo")

        return _resp(
            500,
            {"error": str(error), "tiempos": tiempos},
        )