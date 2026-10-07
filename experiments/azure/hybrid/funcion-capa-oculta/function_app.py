import azure.functions as func
import json

app = func.FunctionApp()

def _distribuir_units(units: int, split_parts: int) -> list[int]:
    """
    Reparte 'units' en 'split_parts' partes lo más uniforme posible.
    Ej.: units=10, parts=3 -> [4,3,3]
    """
    base = units // split_parts
    resto = units % split_parts
    return [base + (1 if i < resto else 0) for i in range(split_parts)]

@app.route(route="http_trigger_capa_oculta", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
def http_trigger_capa_oculta(req: func.HttpRequest) -> func.HttpResponse:
    try:
        req_body = req.get_json()

        # Parámetros clásicos (modo Dense normal)
        units = req_body.get("units")
        activation = req_body.get("activation")
        name = req_body.get("name")

        # Parámetros de particionado ("vertical" en tu vocabulario)
        split_parts = req_body.get("split_parts", 1)           # nº subcapas
        split_mode  = req_body.get("split_mode", "units")      # por ahora: "units"
        recombine   = req_body.get("recombine", "concat")      # por ahora: "concat"
        part_units  = req_body.get("part_units", None)         # opcional: lista explícita

        # Validaciones mínimas
        if units is None or activation is None or name is None:
            return func.HttpResponse(
                json.dumps({"error": "Faltan parámetros obligatorios: 'units', 'activation', 'name'."}, ensure_ascii=False),
                status_code=400,
                mimetype="application/json"
            )

        try:
            units = int(units)
            split_parts = int(split_parts)
        except Exception:
            return func.HttpResponse(
                json.dumps({"error": "'units' y 'split_parts' deben ser enteros."}, ensure_ascii=False),
                status_code=400,
                mimetype="application/json"
            )

        if units <= 0:
            return func.HttpResponse(
                json.dumps({"error": "'units' debe ser > 0."}, ensure_ascii=False),
                status_code=400,
                mimetype="application/json"
            )

        if split_parts < 1:
            return func.HttpResponse(
                json.dumps({"error": "'split_parts' debe ser >= 1."}, ensure_ascii=False),
                status_code=400,
                mimetype="application/json"
            )

        # Import pesado SOLO cuando ya has validado lo básico (evita romper el indexing)
        from tensorflow import keras
        Dense = keras.layers.Dense

        # ─────────────────────────────────────────────────────────────
        # 1) MODO NORMAL (sin partición): devuelvo Dense como hasta ahora
        # ─────────────────────────────────────────────────────────────
        if split_parts == 1:
            layer = Dense(units=units, activation=activation, name=name)
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

        # ─────────────────────────────────────────────────────────────
        # 2) MODO PARTIDO (Camino A): devuelvo spec compuesto SplitDense
        #    - split_mode="units" (partición por neuronas/unidades)
        #    - recombine="concat" (concatenación eje -1)
        # ─────────────────────────────────────────────────────────────
        if split_mode != "units":
            return func.HttpResponse(
                json.dumps({"error": "Por ahora solo se soporta split_mode='units'."}, ensure_ascii=False),
                status_code=400,
                mimetype="application/json"
            )

        if recombine != "concat":
            return func.HttpResponse(
                json.dumps({"error": "Por ahora solo se soporta recombine='concat'."}, ensure_ascii=False),
                status_code=400,
                mimetype="application/json"
            )

        # Units por subcapa: o bien explícito, o bien reparto uniforme
        if part_units is not None:
            if not isinstance(part_units, list) or len(part_units) != split_parts:
                return func.HttpResponse(
                    json.dumps({"error": "'part_units' debe ser una lista con longitud == split_parts."}, ensure_ascii=False),
                    status_code=400,
                    mimetype="application/json"
                )
            try:
                part_units = [int(u) for u in part_units]
            except Exception:
                return func.HttpResponse(
                    json.dumps({"error": "'part_units' debe contener enteros."}, ensure_ascii=False),
                    status_code=400,
                    mimetype="application/json"
                )

            if any(u <= 0 for u in part_units):
                return func.HttpResponse(
                    json.dumps({"error": "Todos los valores de 'part_units' deben ser > 0."}, ensure_ascii=False),
                    status_code=400,
                    mimetype="application/json"
                )

            if sum(part_units) != units:
                return func.HttpResponse(
                    json.dumps({"error": "La suma de 'part_units' debe ser igual a 'units'."}, ensure_ascii=False),
                    status_code=400,
                    mimetype="application/json"
                )
        else:
            part_units = _distribuir_units(units, split_parts)

        # Specs de subcapas Dense
        parts = []
        for i, u in enumerate(part_units, start=1):
            sub_name = f"{name}__part{i}"
            sub_layer = Dense(units=u, activation=activation, name=sub_name)
            parts.append({
                "module": "keras.layers",
                "class_name": "Dense",
                "config": sub_layer.get_config()
            })

        composite_spec = {
            "module": "custom.layers",
            "class_name": "SplitDense",
            "config": {
                "name": name,
                "split_mode": "units",
                "recombine": "concat",
                "axis": -1,
                "units_total": units,
                "parts": parts
            }
        }

        return func.HttpResponse(
            json.dumps(composite_spec, ensure_ascii=False, indent=2),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        return func.HttpResponse(
            json.dumps({"error": str(e)}, ensure_ascii=False),
            status_code=400,
            mimetype="application/json"
        )
