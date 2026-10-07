# Llamada a la función de warm-up
$URL = "<WARMUP_FUNCTION_URL>"
$body = @{} | ConvertTo-Json -Compress
(Invoke-WebRequest -Uri $URL -Method POST -Body $body -ContentType "application/json" -UseBasicParsing).Content



# Llamada a la función de datos
$URL = "<DATA_FUNCTION_URL>"
(Invoke-WebRequest -Uri "$URL/?n=10000" -UseBasicParsing).Content


# Llamada a la funcion del contenedor
# Capa de entrada
$URL = "<TENSORFLOW_CONTAINER_URL>"
$body = @{ units = 8; input_shape = @(2); activation = "relu"; name = "capa_entrada_dense" } | ConvertTo-Json -Compress
(Invoke-WebRequest -Uri "$URL/capa-entrada" -Method POST -Body $body -ContentType "application/json" -UseBasicParsing).Content

# Capa oculta
$URL = "<TENSORFLOW_CONTAINER_URL>"
$body = @{ units = 16; activation = "relu"; name = "oculta_1" } | ConvertTo-Json -Compress
(Invoke-WebRequest -Uri "$URL/capa-oculta" -Method POST -Body $body -ContentType "application/json" -UseBasicParsing).Content

# Capa de salida
$URL = "<TENSORFLOW_CONTAINER_URL>"
$body = @{ units=1; activation="linear"; name="capa_salida_dense" } | ConvertTo-Json -Compress
(Invoke-WebRequest -Uri "$URL/capa-salida" -Method POST -Body $body -ContentType "application/json" -UseBasicParsing).Content



# Llamada a la función de la capa de entrada
$URL = "<INPUT_LAYER_FUNCTION_URL>"
$body = @{ units = 8; input_shape = @(2); activation = "relu"; name = "capa_entrada_dense" } | ConvertTo-Json -Compress
(Invoke-WebRequest -Uri $URL -Method POST -Body $body -ContentType "application/json" -UseBasicParsing).Content



# Llamada a la función de la capa oculta
$URL = "<HIDDEN_LAYER_FUNCTION_URL>"
$body = @{ units = 16; activation = "relu"; name = "oculta_1" } | ConvertTo-Json -Compress
(Invoke-WebRequest -Uri $URL -Method POST -Body $body -ContentType "application/json" -UseBasicParsing).Content



# Llamada a la función de la capa de salida
$URL = "<OUTPUT_LAYER_FUNCTION_URL>"
$body = @{ units=1; activation="linear"; name="capa_salida_dense" } | ConvertTo-Json -Compress
(Invoke-WebRequest -Uri $URL -Method POST -Body $body -ContentType "application/json" -UseBasicParsing).Content



# Llamada a la funcion del modelo
$URL = "<MODEL_FUNCTION_URL>"
$body = @{ num_capas_ocultas = 3 } | ConvertTo-Json -Compress
(Invoke-WebRequest -Uri $URL -Method POST -Body $body -ContentType "application/json" -UseBasicParsing).Content



# Llamada a la funcion de entrenamiento
Write-Output "#################################################################"
Write-Output "ENTRENAMIENTO"
Write-Output "#################################################################"
$URL = "<TRAINING_FUNCTION_URL>"
$body = @{ total_epochs=5; epoch_inicio=0; block_size=5; batch_size=5000 } | ConvertTo-Json -Compress
(Invoke-WebRequest -Uri $URL -Method POST -Body $body -ContentType "application/json" -UseBasicParsing).Content



# Llamada a la funcion de prediccion
Write-Output "#################################################################"
Write-Output "PREDICCION"
Write-Output "#################################################################"
$URL = "<PREDICTION_FUNCTION_URL>"
$body = @{ entrada = @(1.0, 2.0) } | ConvertTo-Json -Compress
(Invoke-WebRequest -Uri $URL -Method POST -Body $body -ContentType "application/json" -UseBasicParsing).Content


$URL = "<PREDICTION_FUNCTION_URL>"
$body = @{ entrada = @(@(1.0,2.0), @(2.0,3.0), @(3.0,4.0)) } | ConvertTo-Json -Compress
(Invoke-WebRequest -Uri $URL -Method POST -Body $body -ContentType "application/json" -UseBasicParsing).Content






$CONT = "<TENSORFLOW_CONTAINER_URL>"
$body = @{
  model_base64 = "<.h5 en base64>"
  X = @(@(1,2), @(3,4))
  Y = @(@(3), @(7))
  optimizer="sgd"; loss="mean_squared_error"; metrics=@("mae","mse","mape")
  epochs=1; initial_epoch=0
} | ConvertTo-Json -Depth 6
Invoke-RestMethod "$CONT/train-model" -Method POST -Body $body -ContentType "application/json"
# Debe devolver "metrics" con loss/mae/mse/mape
