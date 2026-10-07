# Hybrid Architecture for Neural Network Execution in Serverless Environments

This repository contains the experimental source code associated with the scientific work:

> **Hybrid Architecture for Neural Network Execution in Serverless Environments**

The repository includes the implementations used to compare monolithic, segmented and hybrid neural-network execution in serverless environments.

The experimental implementations cover:

- AWS monolithic architecture.
- AWS segmented architecture.
- Azure monolithic architecture.
- Azure segmented architecture.
- Azure hybrid architecture.

The hybrid architecture is the main contribution of this repository. The monolithic and segmented implementations are included as baseline architectures for comparison and reproducibility.

## Repository structure

```text
.
├── experiments/
│   ├── aws/
│   │   ├── monolithic/
│   │   └── segmented/
│   └── azure/
│       ├── monolithic/
│       ├── segmented/
│       └── hybrid/
├── .gitignore
└── README.md
```

## Architectures

### AWS monolithic

`experiments/aws/monolithic/`

Contains the AWS Lambda implementation of the monolithic architecture used as a reference baseline.

Main components include:

- Training function.
- Prediction function.
- Warm-up function.
- TensorFlow container function.
- AWS SAM deployment templates.
- PowerShell scripts used to invoke the deployed functions.

### AWS segmented

`experiments/aws/segmented/`

Contains the AWS Lambda implementation in which the neural network is divided into separate serverless functions.

Main components include:

- Data generation function.
- Input-layer function.
- Hidden-layer function.
- Output-layer function.
- Model assembly function.
- Training function.
- Prediction function.
- Warm-up function.
- TensorFlow container function.
- AWS SAM deployment templates.
- PowerShell scripts used to invoke the deployed functions.

### Azure monolithic

`experiments/azure/monolithic/`

Contains the Azure Functions implementation of the monolithic baseline.

Main components include:

- Training function.
- Prediction function.
- Warm-up function.
- PowerShell scripts used to invoke the deployed functions.

### Azure segmented

`experiments/azure/segmented/`

Contains the fully segmented Azure Functions implementation.

Main components include:

- Data generation function.
- Input-layer function.
- Hidden-layer function.
- Output-layer function.
- Model assembly function.
- Training function.
- Prediction function.
- Warm-up function.
- PowerShell scripts used to invoke the deployed functions.

### Azure hybrid

`experiments/azure/hybrid/`

Contains the hybrid architecture evaluated in the associated research.

The hybrid approach combines serverless orchestration with a segmented neural-network execution model while reducing some of the overhead introduced by a fully segmented architecture.

Main components include:

- Data generation function.
- Input-layer function.
- Hidden-layer function.
- Output-layer function.
- Model assembly function.
- Training function.
- Prediction function.
- Warm-up function.
- Local model verification utilities.
- PowerShell scripts used to invoke the deployed functions.

## Experimental scope

The source code was developed to evaluate the behavior of different neural-network execution architectures in serverless environments.

The experiments consider aspects such as:

- Execution time.
- Memory consumption.
- CPU usage.
- Temporary disk usage.
- Training workload.
- Prediction workload.
- Input data volume.
- Neural-network depth.
- Batch size.
- Number of epochs.
- Serverless execution limits.
- Overhead introduced by function decomposition.
- Differences between monolithic, segmented and hybrid approaches.

The repository is intended to support inspection of the implementation and facilitate reproduction of the experimental procedure.

## Requirements

The implementations use Python 3.11.

Depending on the selected platform, the following tools may be required.

### AWS

- AWS account.
- AWS CLI.
- AWS SAM CLI.
- Docker Desktop or another Docker-compatible runtime.
- PowerShell.
- Permissions to create and invoke AWS Lambda functions.
- Permissions to create and access Amazon S3 resources.
- Permissions to create container images and Amazon ECR repositories.

### Azure

- Microsoft Azure account.
- Azure CLI.
- Azure Functions Core Tools.
- Python 3.11.
- PowerShell.
- Azure Storage account.
- Permissions to create and configure Azure Function Apps.

The dependencies required by each function are declared in the corresponding `requirements.txt` file.

## Configuration

Real cloud endpoints, account identifiers, access keys and connection strings have been removed from the public repository.

Before deploying or invoking the functions, replace the placeholder values with resources from your own cloud account.

Examples of placeholders used in the repository include:

```text
<TENSORFLOW_CONTAINER_URL>
<DATA_FUNCTION_URL>
<INPUT_LAYER_FUNCTION_URL>
<HIDDEN_LAYER_FUNCTION_URL>
<OUTPUT_LAYER_FUNCTION_URL>
<MODEL_FUNCTION_URL>
<TRAINING_FUNCTION_URL>
<PREDICTION_FUNCTION_URL>
<WARMUP_FUNCTION_URL>
```

Do not commit credentials, storage keys, connection strings or private endpoints.

### Azure Storage configuration

The Azure baseline implementations and the hybrid implementation use different environment variable names for the storage connection string.

Baseline Azure implementations:

```text
AZURE_STORAGE_CONNECTION_STRING
```

Hybrid Azure implementation:

```text
STORAGE_CONNECTION_STRING
```

Configure this value in the Azure Function App settings or in a local `local.settings.json` file when running locally.

The `local.settings.json` file must not be committed because it may contain credentials.

Example:

```json
{
  "IsEncrypted": false,
  "Values": {
    "AzureWebJobsStorage": "UseDevelopmentStorage=true",
    "FUNCTIONS_WORKER_RUNTIME": "python",
    "STORAGE_CONNECTION_STRING": "<YOUR_AZURE_STORAGE_CONNECTION_STRING>"
  }
}
```

### AWS configuration

AWS function URLs and resource names must be configured in the corresponding AWS SAM templates or deployment environment.

Container-specific `samconfig.toml` files containing account-specific Amazon ECR configuration are intentionally excluded.

A deployment configuration can be generated with:

```bash
sam build
sam deploy --guided
```

## Running the experiments

PowerShell scripts named `Llamadas_Funciones.ps1` are included in the architecture directories.

Before running an experiment:

1. Deploy all functions required by the selected architecture.
2. Replace the endpoint placeholders with the corresponding function URLs.
3. Configure the required storage resources.
4. Review the request parameters used by the experiment.
5. Verify memory, timeout and deployment settings.
6. Ensure that previous models or temporary files do not interfere with the execution.

Example:

```powershell
$URL = "<TRAINING_FUNCTION_URL>"

Invoke-RestMethod `
    -Uri $URL `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

The scripts contain requests representative of the experimental workloads. Review all parameters before execution because cloud invocations may generate costs.

## Experimental parameters

Depending on the experiment, the request bodies and configuration files may include parameters such as:

- Number of training samples.
- Number of epochs.
- Batch size.
- Number of hidden layers.
- Number of neurons.
- Initial epoch.
- Training block size.
- Input values used for prediction.

The exact parameter combinations used in the article should be interpreted together with the methodology and experimental sections of the associated manuscript.

## Reproducibility considerations

The repository provides the source code and deployment configuration used to implement the experiments. Exact numerical reproduction may still vary because the execution environment is managed by the cloud provider.

Potential sources of variation include:

- Cold and warm function invocations.
- Cloud-provider scheduling.
- Available CPU capacity.
- Network latency.
- Runtime and dependency updates.
- Regional infrastructure changes.
- Function concurrency.
- Temporary storage state.
- Provider-specific execution limits.
- TensorFlow initialization and nondeterministic operations.

Where explicitly defined in the source code, fixed NumPy seeds are preserved. However, identical timing, CPU and memory measurements cannot be guaranteed across different cloud executions.

The repository should therefore be considered reproducible at the architectural and procedural level, while performance measurements remain dependent on the serverless execution environment.

## Security

This repository does not include active cloud credentials or intentionally expose deployed endpoints.

Users must provide their own:

- AWS or Azure credentials.
- Storage resources.
- Function URLs.
- Resource identifiers.
- Access policies.
- Connection strings.

Never commit:

- AWS access keys.
- Azure Storage keys.
- Connection strings.
- Private endpoints.
- `.env` files.
- `local.settings.json`.
- Private certificates.
- Account-specific deployment files.

The code is research-oriented experimental software and is not intended to be deployed directly as a production service.

## Cost warning

Deploying and executing these experiments on AWS or Azure may generate charges.

Costs may be associated with:

- Serverless function invocations.
- Execution duration.
- Allocated memory.
- Storage operations.
- Container registries.
- Network transfer.
- Logging and monitoring.
- Resources left deployed after the experiments.

Users are responsible for reviewing current provider pricing, monitoring resource consumption and deleting deployed resources after completing the experiments.

## Limitations

This repository is not intended to provide:

- A production Machine Learning platform.
- A secure public prediction service.
- A general-purpose neural-network framework.
- A cost-optimized deployment.
- A provider-independent abstraction layer.

Some configuration values and deployment steps must be adapted manually to the user's cloud environment.

## Citation

When using this repository, please cite the associated research:

> **Hybrid Architecture for Neural Network Execution in Serverless Environments**

Complete bibliographic information will be added when the article is formally published.

A `CITATION.cff` file will be included to provide machine-readable citation metadata.

## License

The source code will be distributed under the license included in the `LICENSE` file.

## Authors

Author information will be completed using the definitive metadata of the associated article.

## Contact

Questions regarding the implementation and reproduction of the experiments can be submitted through the GitHub repository issue tracker.
