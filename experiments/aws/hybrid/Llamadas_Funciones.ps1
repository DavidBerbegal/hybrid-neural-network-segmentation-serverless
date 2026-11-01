# Diagnosticos de AWS hibrido.
# Ejecutar los bloques necesarios de forma individual.
# No es el script definitivo de resultados experimentales.

$AwsProfile = "default"
$AwsRegion = "eu-west-3"


# ============================================================
# CONFIGURACION: ejecutar antes de los bloques de diagnostico
# ============================================================

function Get-StackOutput {
    param(
        [string]$StackName,
        [string]$OutputKey
    )

    $value = aws cloudformation describe-stacks `
        --stack-name $StackName `
        --profile $AwsProfile `
        --region $AwsRegion `
        --query "Stacks[0].Outputs[?OutputKey=='$OutputKey'].OutputValue" `
        --output text

    if (
        $LASTEXITCODE -ne 0 -or
        [string]::IsNullOrWhiteSpace($value) -or
        $value.Trim() -eq "None"
    ) {
        throw "No se pudo obtener $OutputKey del stack $StackName."
    }

    return $value.Trim().TrimEnd("/")
}

function Invoke-JsonPost {
    param(
        [string]$Url,
        [hashtable]$Payload
    )

    Invoke-RestMethod `
        -Uri $Url `
        -Method Post `
        -Body ($Payload | ConvertTo-Json -Depth 30 -Compress) `
        -ContentType "application/json"
}

$WarmUpUrl = Get-StackOutput "funcion-warm-up" "WarmUpUrl"
$DatosUrl = Get-StackOutput "funcion-datos" "FunctionUrl"
$ContenedorUrl = Get-StackOutput "tf-container" "FunctionUrl"
$EntradaUrl = Get-StackOutput "funcion-capa-entrada" "FunctionUrl"
$OcultaUrl = Get-StackOutput "funcion-capa-oculta" "FunctionUrl"
$SalidaUrl = Get-StackOutput "funcion-capa-salida" "FunctionUrl"
$ModeloUrl = Get-StackOutput "funcion-modelo" "FunctionUrl"
$EntrenamientoUrl = Get-StackOutput "funcion-entrenamiento" "FunctionUrl"
$PrediccionUrl = Get-StackOutput "funcion-prediccion" "FunctionUrl"


# ============================================================
# WARM-UP
# Sobrescribe el CSV de S3 con 1000 muestras.
# ============================================================

Invoke-JsonPost "$WarmUpUrl/" @{} |
    ConvertTo-Json -Depth 20


# ============================================================
# DATOS
# Sobrescribe el CSV de S3 con 10000 muestras.
# ============================================================

Invoke-RestMethod `
    -Uri "$DatosUrl/?n=10000" `
    -Method Get |
    ConvertTo-Json -Depth 20


# ============================================================
# CONTENEDOR: CAPA DE ENTRADA
# Especificacion independiente; no modifica el modelo de S3.
# ============================================================

Invoke-JsonPost "$ContenedorUrl/capa-entrada" @{
    units = 8
    input_shape = @(2)
    activation = "relu"
    name = "capa_entrada_dense"
} | ConvertTo-Json -Depth 20


# ============================================================
# CONTENEDOR: CAPA OCULTA
# ============================================================

Invoke-JsonPost "$ContenedorUrl/capa-oculta" @{
    units = 16
    activation = "relu"
    name = "oculta_1"
} | ConvertTo-Json -Depth 20


# ============================================================
# CONTENEDOR: CAPA DE SALIDA
# ============================================================

Invoke-JsonPost "$ContenedorUrl/capa-salida" @{
    units = 1
    activation = "linear"
    name = "capa_salida_dense"
} | ConvertTo-Json -Depth 20


# ============================================================
# LAMBDA ZIP: CAPA DE ENTRADA
# ============================================================

Invoke-JsonPost "$EntradaUrl/" @{
    units = 8
    input_shape = @(2)
    activation = "relu"
    name = "capa_entrada_dense"
} | ConvertTo-Json -Depth 20


# ============================================================
# LAMBDA ZIP: CAPA OCULTA
# ============================================================

Invoke-JsonPost "$OcultaUrl/" @{
    units = 16
    activation = "relu"
    name = "oculta_1"
} | ConvertTo-Json -Depth 20


# ============================================================
# LAMBDA ZIP: CAPA DE SALIDA
# ============================================================

Invoke-JsonPost "$SalidaUrl/" @{
    units = 1
    activation = "linear"
    name = "capa_salida_dense"
} | ConvertTo-Json -Depth 20


# ============================================================
# MODELO
# Sobrescribe modelo_sin_entrenar.h5 en S3.
# Conserva el ejemplo original: tres capas ocultas sin particion.
# El modelo utiliza internamente entrada 5, ocultas 4 y salida 1.
# Las especificaciones independientes anteriores no se reutilizan.
# ============================================================

Invoke-JsonPost "$ModeloUrl/" @{
    num_capas_ocultas = 3
} | ConvertTo-Json -Depth 20


# ============================================================
# ENTRENAMIENTO
# Prioriza modelo_entrenado.h5 si existe.
# batch_size selecciona filas del CSV; no es el minibatch de fit.
# Sobrescribe modelo_entrenado.h5.
# ============================================================

Write-Output "ENTRENAMIENTO"

Invoke-JsonPost "$EntrenamientoUrl/" @{
    total_epochs = 5
    epoch_inicio = 0
    block_size = 5
    batch_size = 5000
} | ConvertTo-Json -Depth 20


# ============================================================
# PREDICCION: UNA MUESTRA
# ============================================================

Write-Output "PREDICCION INDIVIDUAL"

Invoke-JsonPost "$PrediccionUrl/" @{
    entrada = @(1.0, 2.0)
} | ConvertTo-Json -Depth 20


# ============================================================
# PREDICCION: VARIAS MUESTRAS
# La API devuelve solo el primer resultado, como Azure hibrido.
# ============================================================

Write-Output "PREDICCION MULTIPLE"

Invoke-JsonPost "$PrediccionUrl/" @{
    entrada = @(
        @(1.0, 2.0)
        @(2.0, 3.0)
        @(3.0, 4.0)
    )
} | ConvertTo-Json -Depth 20


# ============================================================
# CONTENEDOR: ENTRENAMIENTO AISLADO
# Descarga el modelo sin entrenar de S3.
# El resultado permanece en memoria; no se sube a S3.
# ============================================================

$configJson = aws lambda get-function-configuration `
    --function-name funcion-entrenamiento `
    --profile $AwsProfile `
    --region $AwsRegion `
    --query "Environment.Variables" `
    --output json

if ($LASTEXITCODE -ne 0) {
    throw "No se pudo obtener la configuracion de almacenamiento."
}

$storageConfig = $configJson | ConvertFrom-Json

$diagnosticoDir = Join-Path $env:TEMP "aws-hybrid-diagnostics"
New-Item -ItemType Directory -Path $diagnosticoDir -Force |
    Out-Null

$modeloPath = Join-Path $diagnosticoDir "modelo_sin_entrenar.h5"
$s3Source = "s3://$($storageConfig.S3_BUCKET_NAME)/$($storageConfig.UNTRAINED_KEY)"

aws s3 cp `
    $s3Source `
    $modeloPath `
    --profile $AwsProfile `
    --region $AwsRegion

if ($LASTEXITCODE -ne 0) {
    throw "No se pudo descargar el modelo sin entrenar."
}

$modelBase64 = [Convert]::ToBase64String(
    [System.IO.File]::ReadAllBytes($modeloPath)
)

$entrenamientoDirecto = Invoke-JsonPost "$ContenedorUrl/train-model" @{
    model_base64 = $modelBase64
    X = @(@(1, 2), @(3, 4))
    Y = @(@(3), @(7))
    optimizer = "sgd"
    loss = "mean_squared_error"
    metrics = @("mae", "mse", "mape")
    epochs = 1
    initial_epoch = 0
}

$entrenamientoDirecto.metrics | ConvertTo-Json
$entrenamientoDirecto.tiempos | ConvertTo-Json