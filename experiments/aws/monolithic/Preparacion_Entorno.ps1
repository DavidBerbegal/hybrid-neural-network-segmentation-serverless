# Instalar docker desktop
# Instalar "AWS Toolkit" en VS Code
# sam build
# sam deploy / sam deploy --guided

# Obtener URL de función
aws lambda get-function-url-config `
  --function-name funcion-warm-up-monolitica `
  --region eu-west-3 `
  --query FunctionUrl `
  --output text