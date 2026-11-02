# Experimental Results

This directory contains the experimental results supporting the manuscript **“Hybrid Architecture for Neural Network Execution in Serverless Environments”**.

The study compares three neural network execution configurations on **Azure Functions** and **AWS Lambda**:

- `monolitica`: monolithic implementation, used as an external reference for operational limits.
- `segmentada`: reference inter-layer partitioned architecture.
- `hibrida`: hybrid architecture combining inter-layer partitioning and intra-layer split-recombine operations.

The primary comparison is between `segmentada` and `hibrida`. The monolithic implementation is included to characterize operational limitations.

## Files

| File | Experimental scenario | Rows |
| --- | --- | ---: |
| `Graficas_Test_01.csv` | Increasing dataset size, from 20,000 to 700,000 records (35 iterations per configuration) | 210 |
| `Graficas_Test_02.csv` | Sustained load, up to 200 planned iterations per configuration | 1,200 |
| `Graficas_Test_03.csv` | Deep chaining with 20 hidden layers (20 iterations per configuration) | 120 |
| `Graficas_Test_04.csv` | Large batch size of 300,000 (20 iterations per configuration) | 120 |
| `Graficas_Test_05.csv` | 30 hidden layers and 20 configured epochs (20 iterations per configuration) | 120 |
| **Total** | | **1,770** |

Each CSV contains rows for the three configurations on both providers. **A row does not necessarily represent a completed execution**: the files also preserve failed and missing/incomplete observations.

## CSV format

- Encoding: UTF-8.
- Field separator: semicolon (`;`).
- The `*_normalizado` columns contain normalized values exported from the Excel analysis sheets and may use a **comma as the decimal separator**. Read these fields with a locale-aware parser or explicitly replace decimal commas when processing them.
- Other numeric-looking fields are preserved as exported. **Do not assume that a field without the `_normalizado` suffix can be interpreted directly as a decimal measurement**: Excel formatting and locale conversion may affect their representation. Check the paired normalized field before using a measurement in a calculation.
- Empty cells indicate unavailable or unrecorded information and must not be converted to zero.

For example, a `tiempo_s_normalizado` value of `23,984` denotes **23.984 seconds**, not 23,984 seconds. The unnormalized `tiempo_s` field may display `23984` for the same observation.

## Column reference

**Execution identification**

- `pivot`: label used in the original spreadsheet for aggregation.
- `proveedor`: cloud provider (`AZURE` or `AWS`).
- `iteracion`: test iteration number.
- `n_datos`: configured dataset size / number of records in the input file.
- `arquitectura`: execution configuration (`monolitica`, `segmentada`, `hibrida`).

**Execution status and error information**

- `status_code`: recorded HTTP response code, when available.
- `error`: error message, when present. Infrastructure hostnames have been removed from the publicly shared version.
- `error_type`: recorded error classification, when available.
- `is_timeout`, `is_oom`, `is_cpu_high`: classification fields included in the exported dataset. A blank entry is not evidence that an event did or did not occur; consult the status and error message.
- `timeout_s`: timeout-related recorded field; the corresponding normalized column is `timeout_s_normalizado`.

**Execution and resource measurements**

- `tiempo_s`: recorded execution time; consult `tiempo_s_normalizado` for the decimal representation in seconds.
- `tiempo_preparacion`: recorded preparation time; consult `tiempo_preparacion_normalizado` for the corresponding normalized value.
- `cpu_inicial_percent`, `cpu_final_percent`: CPU utilization measurements, in percent.
- `memoria_inicial_mb`, `memoria_final_mb`, `memoria_consumida_mb`, `memoria_batch_max_mb`: recorded memory measurements, in MB.
- `disk_inicial_mb`, `disk_final_mb`, `disk_consumido_mb`: recorded disk measurements, in MB.

**Model training metrics**

- `loss`, `mae`, `mse`, `mape`: training/evaluation metrics as recorded in the test outputs.

A normalized field with suffix `_normalizado` is also provided for each applicable numerical measurement. The original export contains **43 columns**.

## Interpretation notes

1. **Success and failure:** analyse HTTP response codes and error messages before aggregating execution times or resource measurements. Failed executions must not be treated as successful runs. A timeout is not, by itself, evidence of an out-of-memory condition.
2. **Test 02 interruption:** the AWS hybrid test recorded 179 successful executions, followed by an HTTP client timeout at iteration 180 (600-second wait). The remaining 20 planned iterations were not executed. Their presence in the dataset must not be interpreted as completed measurements.
3. **AWS hybrid dataset processing:** in the evaluated AWS hybrid implementation, each invocation uses the first 10,000 records from the CSV file for training. Thus, increases in `n_datos` do not imply a corresponding increase in the number of training samples processed. This applies to the AWS hybrid scenarios described in the manuscript and is essential when interpreting cross-configuration comparisons.
4. **AWS metric scope:** some AWS hybrid CPU and memory measurements are reported for the proxy/training function, while other measurements originate from the training container. These measurements are not interchangeable. Consult the manuscript for the instrumentation scope applicable to each test.
5. **Provider comparison:** execution time and resource figures may reflect different implementation and instrumentation scopes. The dataset supports the comparisons reported in the manuscript but should not be treated as proof that intra-layer partitioning alone caused every observed difference.

## Source and reproducibility

These CSV files were exported from the `Datos` worksheets of the five experiment workbooks used to prepare the manuscript's figures. The public versions preserve the experimental records and measurements while removing AWS Lambda endpoint hostnames from error messages.

The associated implementation is available at:

https://github.com/DavidBerbegal/hybrid-neural-network-segmentation-serverless

Refer to the manuscript's methodology and results sections for experimental configurations, training settings, measurement scopes, and discussion of the operational failures.
