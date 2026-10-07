# Llamada a la función de warm-up
$URL = "<WARM_UP_URL>"
$body = @{} | ConvertTo-Json -Compress
(Invoke-WebRequest -Uri $URL -Method POST -Body $body -ContentType "application/json" -UseBasicParsing).Content



# Llamada a la funcion de entrenamiento
Write-Output "#################################################################"
Write-Output "MODELO + ENTRENAMIENTO"
Write-Output "#################################################################"
$URL = "<TRAINING_URL>"
$Body = @{
    total_epochs      = 10
    epoch_inicio      = 0
    block_size        = 50
    num_capas_ocultas = 3
    n_datos           = 1000
} | ConvertTo-Json -Depth 3

Invoke-RestMethod -Uri $URL -Method POST -Body $Body -ContentType "application/json"


Write-Output "#################################################################"
Write-Output "MODELO + ENTRENAMIENTO"
Write-Output "#################################################################"
$URL = "<TRAINING_URL>"
$r = Invoke-RestMethod -Uri $URL -Method POST -Body $Body -ContentType "application/json"

# Ver todo el JSON real
$r | ConvertTo-Json -Depth 6

# Ver solo las métricas del batch
$r.entrenamiento_batch[0].metricas | Format-List *

# O como JSON
$r.entrenamiento_batch[0].metricas | ConvertTo-Json -Depth 3




# Llamada a la funcion de prediccion
Write-Output "#################################################################"
Write-Output "PREDICCION"
Write-Output "#################################################################"
$URL = "<PREDICTION_URL>"
$body = @{ entrada = @(1.0, 2.0) } | ConvertTo-Json -Compress
(Invoke-WebRequest -Uri $URL -Method POST -Body $body -ContentType "application/json" -UseBasicParsing).Content


$URL = "<PREDICTION_URL>"
$body = @{ entrada = @(@(1.0,2.0), @(2.0,3.0), @(3.0,4.0)) } | ConvertTo-Json -Compress
(Invoke-WebRequest -Uri $URL -Method POST -Body $body -ContentType "application/json" -UseBasicParsing).Content
