# Llamada a la función de warm-up
(Invoke-WebRequest -Uri "http://localhost:7071/api/http_trigger_warm_up" -UseBasicParsing).Content
(Invoke-WebRequest -Uri "<WARMUP_FUNCTION_URL>" -UseBasicParsing).Content



# Llamada a la función de datos
(Invoke-WebRequest -Uri "http://localhost:7071/api/http_trigger_datos?n=500000" -UseBasicParsing).Content
(Invoke-WebRequest -Uri "<DATA_FUNCTION_URL>?n=500000" -UseBasicParsing).Content


Invoke-WebRequest -Uri "http://localhost:7071/api/http_trigger_datos?n=500000" -UseBasicParsing
Invoke-WebRequest -Uri "<DATA_FUNCTION_URL>?n=10000" -UseBasicParsing


#$resp = Invoke-WebRequest -Uri "http://localhost:7071/api/http_trigger_datos?n=500000" -UseBasicParsing
$resp = Invoke-WebRequest -Uri "<DATA_FUNCTION_URL>?n=500000" -UseBasicParsing
$data = $resp.Content | ConvertFrom-Json

# Muestra sólo el objeto 'recursos'
$data.recursos



# Llamada a la función de la capa de entrada
#Invoke-RestMethod -Uri "http://localhost:7071/api/http_trigger_capa_entrada" `
Invoke-RestMethod -Uri "<INPUT_LAYER_FUNCTION_URL>" `
  -Method POST `
  -Body (@{ units = 5; input_shape = @(2); activation = "relu"; name = "entrada" } | ConvertTo-Json) `
  -ContentType "application/json" | ConvertTo-Json -Depth 10



# Llamada a la función de la capa oculta
Invoke-RestMethod -Uri "<HIDDEN_LAYER_FUNCTION_URL>" `
  -Method POST `
  -Body (@{ units = 8; activation = "relu"; name = "capa_oculta_dense" } | ConvertTo-Json) `
  -ContentType "application/json" | ConvertTo-Json -Depth 10



# Llamada a la función de la capa de salida
Invoke-RestMethod -Uri "<OUTPUT_LAYER_FUNCTION_URL>" `
  -Method POST `
  -Body (@{ units = 1; activation = "sigmoid"; name = "capa_salida_dense" } | ConvertTo-Json) `
  -ContentType "application/json" | ConvertTo-Json -Depth 10



# Llamada a la funcion del modelo
try {
    #Invoke-RestMethod -Uri "http://localhost:7071/api/http_trigger_modelo" `
    Invoke-RestMethod -Uri "<MODEL_FUNCTION_URL>" `
      -Method POST `
      -Body (@{ num_capas_ocultas = 3 } | ConvertTo-Json) `
      -ContentType "application/json" | ConvertTo-Json -Depth 10
} catch {
    Write-Host "Error Body:"
    Write-Host $_.Exception.Response.GetResponseStream() | Get-Content
}


$data = Invoke-RestMethod -Uri "http://localhost:7071/api/http_trigger_modelo" `
#$data = Invoke-RestMethod -Uri "<MODEL_FUNCTION_URL>" `
    -Method POST `
    -Body (@{ num_capas_ocultas = 3 } | ConvertTo-Json) `
    -ContentType "application/json"

# Muestra sólo el objeto 'recursos'
$data.recursos | Format-List



# Llamada a la funcion de entrenamiento
Write-Output "#################################################################"
Write-Output "ENTRENAMIENTO"
Write-Output "#################################################################"

# 1) Configuración general
$uri            = "http://localhost:7071/api/http_trigger_entrenamiento"
#$uri          = "<TRAINING_FUNCTION_URL>"
$total_epochs   = 5
$block_size     = 5
$batch_size     = 1000
$n_datos_total  = 10000
$max_retries    = 5
$csv_file       = "resultados_entrenamiento_segmentado.csv"

# 2) Cabeceras del CSV
$fields = @(
    "batch_num","total_epochs","block_size","batch_size","attempts",
    "status_code","error","tiempo_total_s",
    "memoria_inicial_mb","memoria_final_mb","memoria_consumida_mb",
    "cpu_inicial_percent","cpu_final_percent",
    "disk_inicial_mb","disk_final_mb","disk_consumido_mb",
    "loss","mae","mse","mape"
)

# 3) Prepara/borra CSV anterior
"" | Set-Content -Path $csv_file
$fields -join "," | Add-Content -Path $csv_file

# 4) Bucle de batches
$batch_num = 0
for ($start = 0; $start -lt $n_datos_total; $start += $batch_size) {
    # payload JSON
    $body = @{
        total_epochs = $total_epochs
        epoch_inicio = $start / $batch_size * $block_size
        block_size   = $block_size
        batch_size   = $batch_size
    } | ConvertTo-Json

    $success  = $false
    $attempts = 0

    while (-not $success -and $attempts -lt $max_retries) {
        try {
            Write-Host "Batch $($batch_num+1): intento $($attempts+1)"
            $t_ini = Get-Date

            $response = Invoke-RestMethod -Uri $uri `
                -Method POST `
                -Body $body `
                -ContentType "application/json" `
                -TimeoutSec 300

            $t_total = (Get-Date) - $t_ini

            # Montar fila inicial
            $row = @{
                batch_num             = $batch_num + 1
                total_epochs          = $total_epochs
                block_size            = $block_size
                batch_size            = $batch_size
                attempts              = $attempts + 1
                status_code           = 200
                error                 = ""
                tiempo_total_s        = [math]::Round($t_total.TotalSeconds,2)
                memoria_inicial_mb    = ""
                memoria_final_mb      = ""
                memoria_consumida_mb  = ""
                cpu_inicial_percent   = ""
                cpu_final_percent     = ""
                disk_inicial_mb       = ""
                disk_final_mb         = ""
                disk_consumido_mb     = ""
                loss                  = ""
                mae                   = ""
                mse                   = ""
                mape                  = ""
            }

            # Extraer métricas de recursos
            if ($response.tiempos) {
                $t = $response.tiempos
                $row.memoria_inicial_mb   = $t.memoria_inicial_mb
                $row.memoria_final_mb     = $t.memoria_final_mb
                $row.memoria_consumida_mb = $t.memoria_consumida_mb
                $row.cpu_inicial_percent  = $t.cpu_inicial_percent
                $row.cpu_final_percent    = $t.cpu_final_percent
                $row.disk_inicial_mb      = $t.disk_inicial_mb
                $row.disk_final_mb        = $t.disk_final_mb
                $row.disk_consumido_mb    = $t.disk_consumido_mb
            }

            # Extraer métricas de entrenamiento
            if ($response.metricas_finales) {
                $m = $response.metricas_finales
                $row.loss = $m.loss
                $row.mae  = $m.mae
                $row.mse  = $m.mse
                $row.mape = $m.mape
            }

            $success = $true
            Write-Host "Batch $($batch_num+1) completado con éxito."
        }
        catch {
            # En caso de error, registrar excepción
            $row = @{
                batch_num             = $batch_num + 1
                total_epochs          = $total_epochs
                block_size            = $block_size
                batch_size            = $batch_size
                attempts              = $attempts + 1
                status_code           = $_.Exception.Response.StatusCode.value__
                error                 = $_.Exception.Message.Replace("`n"," ").Replace("`r"," ")
                tiempo_total_s        = ""
                memoria_inicial_mb    = ""
                memoria_final_mb      = ""
                memoria_consumida_mb  = ""
                cpu_inicial_percent   = ""
                cpu_final_percent     = ""
                disk_inicial_mb       = ""
                disk_final_mb         = ""
                disk_consumido_mb     = ""
                loss                  = ""
                mae                   = ""
                mse                   = ""
                mape                  = ""
            }
            Write-Warning "Error en batch $($batch_num+1): $($_.Exception.Message)"
        }

        $attempts++
        if (-not $success -and $attempts -lt $max_retries) {
            Start-Sleep -Seconds 15
            Write-Host "Reintentando batch $($batch_num+1)..."
        }
    }

    # 5) Guardar fila en CSV (asegurando punto decimal)
    $fields | ForEach-Object { $row.$_ = ($row.$_ -as [string]).Replace(",",".") }
    ($fields | ForEach-Object { $row.$_ }) -join "," | Add-Content -Path $csv_file

    if (-not $success) {
        Write-Error "Batch $($batch_num+1) falló tras $max_retries intentos. Abortando."
        break
    }

    $batch_num++
}

Write-Host "=== Entrenamiento segmentado COMPLETADO ==="
Write-Host "Resultados guardados en $csv_file"


$data = Invoke-RestMethod -Uri $uri -Method POST -Body $body -ContentType "application/json"
$data.tiempos | Format-List



# Llamada a la funcion de prediccion
Write-Output "#################################################################"
Write-Output "PREDICCION"
Write-Output "#################################################################"

$body = @{
    entrada = @(
        @(10, 15)
    )
} | ConvertTo-Json

#Invoke-RestMethod -Uri "http://localhost:7071/api/http_trigger_prediccion" `
Invoke-RestMethod -Uri "<PREDICTION_FUNCTION_URL>" `
  -Method POST `
  -Body $body `
  -ContentType "application/json" | ConvertTo-Json -Depth 10