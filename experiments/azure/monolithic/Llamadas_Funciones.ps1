# Llamada a la función de warm-up
(Invoke-WebRequest -Uri "http://localhost:7071/api/http_trigger_warm_up_monolitica" -UseBasicParsing).Content
(Invoke-WebRequest -Uri "<WARMUP_FUNCTION_URL>" -UseBasicParsing).Content



# Llamada a la funcion de entrenamiento
Write-Output "#################################################################"
Write-Output "MODELO + ENTRENAMIENTO"
Write-Output "#################################################################"

# Configuración
#$uri            = "http://localhost:7071/api/http_trigger_entrenamiento_monolitica"
$uri          = "<TRAINING_FUNCTION_URL>"
$total_epochs        = 5
$block_size          = 5
$batch_size          = 1000
$n_datos_total       = 1000
$max_retries         = 5
$num_capas_ocultas   = 3
$csv_file            = "resultados_entrenamiento_monolitica.csv"

# Cabeceras del CSV
$fields = @(
    "batch_num","total_epochs","block_size","batch_size","num_capas_ocultas",
    "attempts","status_code","error","tiempo_total_s",
    "memoria_consumida_mb","disco_consumido_mb",
    "loss","mae","mse","mape"
)

# Crear CSV (sobrescribe si ya existe)
"" | Set-Content -Path $csv_file
$fields -join "," | Add-Content -Path $csv_file

# Bucle de batches
$batch_num = 0
for ($start = 0; $start -lt $n_datos_total; $start += $batch_size) {
    $payload = @{
        total_epochs      = $total_epochs
        epoch_inicio      = 0
        block_size        = $block_size
        n_datos           = $batch_size
        num_capas_ocultas = $num_capas_ocultas
    } | ConvertTo-Json

    $success  = $false
    $attempts = 0

    while (-not $success -and $attempts -lt $max_retries) {
        try {
            Write-Host "Batch $($batch_num+1): intento $($attempts+1)"
            $t_ini    = Get-Date

            $response = Invoke-RestMethod -Uri $uri `
                -Method POST `
                -Body $payload `
                -ContentType "application/json" `
                -TimeoutSec 300

            $t_total = (Get-Date) - $t_ini

            # Base de la fila
            $row = @{
                batch_num             = $batch_num + 1
                total_epochs          = $total_epochs
                block_size            = $block_size
                batch_size            = $batch_size
                num_capas_ocultas     = $num_capas_ocultas
                attempts              = $attempts + 1
                status_code           = 200
                error                 = ""
                tiempo_total_s        = [math]::Round($t_total.TotalSeconds,2)
                memoria_consumida_mb  = ""
                disco_consumido_mb    = ""
                loss                  = ""
                mae                   = ""
                mse                   = ""
                mape                  = ""
            }

            # Extraer métricas de tiempos
            if ($response.tiempos) {
                $t = $response.tiempos
                $row.memoria_consumida_mb = $t.memoria_consumida_mb
                $row.disco_consumido_mb   = $t.disco_consumido_mb
            }

            # Extraer métricas ML
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
            $row = @{
                batch_num             = $batch_num + 1
                total_epochs          = $total_epochs
                block_size            = $block_size
                batch_size            = $batch_size
                num_capas_ocultas     = $num_capas_ocultas
                attempts              = $attempts + 1
                status_code           = $_.Exception.Response.StatusCode.value__
                error                 = $_.Exception.Message.Replace("`n"," ").Replace("`r"," ")
                tiempo_total_s        = ""
                memoria_consumida_mb  = ""
                disco_consumido_mb    = ""
                loss                  = ""
                mae                   = ""
                mse                   = ""
                mape                  = ""
            }
            Write-Warning "Excepción en batch $($batch_num+1): $($_.Exception.Message)"
        }

        $attempts++
        if (-not $success -and $attempts -lt $max_retries) {
            Start-Sleep -Seconds 15
            Write-Host "Reintentando batch $($batch_num+1)..."
        }
    }

    # Escribir la fila en el CSV
    $fields | ForEach-Object { $row.$_ = ($row.$_ -as [string]).Replace(",",".") }
    ($fields | ForEach-Object { $row.$_ }) -join "," | Add-Content -Path $csv_file

    if (-not $success) {
        Write-Error "Batch $($batch_num+1) falló tras $max_retries intentos. Abortando."
        break
    }

    $batch_num++
}

Write-Host "==== Entrenamiento monolítico COMPLETADO ===="
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

#Invoke-RestMethod -Uri "http://localhost:7071/api/http_trigger_prediccion_monolitica" `
Invoke-RestMethod -Uri "<PREDICTION_FUNCTION_URL>" `
  -Method POST `
  -Body $body `
  -ContentType "application/json" | ConvertTo-Json -Depth 10