# Kumo-Anomaly fine-tuning for time-series anomaly detection

Fine-tune NVIDIA's [Kumo-Anomaly](https://huggingface.co/nvidia/Kumo-Anomaly) diffusion model on the public [Kaggle multivariate time-series dataset](https://www.kaggle.com/datasets/drscarlat/time-series) for 10 epochs on one A10 GPU. The recipe runs the fine-tuning example from NVIDIA's [Kumo-TS repository](https://github.com/NVIDIA/Kumo-TS/tree/main/Kumo-Anomaly) at a pinned commit, inside that project's locked `uv` environment.

Kumo-Anomaly learns what normal behavior looks like, so the recipe trains and validates on anomaly-free data only. The dataset has 11 numeric features and a 0/1 anomaly label for each row.

The recipe splits the series in time order into three parts: the first 60% for training, the next 20% for validation, and the last 20% held out as a test set. This recipe does not use that test set. It is reserved for measuring detection performance before and after fine-tuning, which the upstream detection step covers (see Results below). Within the training and validation parts, the recipe scans the series in non-overlapping 100-row blocks and keeps only the blocks that contain no labeled anomalies. It keeps the earliest such blocks up to a fixed count: 500 blocks (50,000 rows) for training and 100 blocks (10,000 rows) for validation. Labels only decide which blocks to keep. The model never sees them.

The recipe reports the upstream training and validation losses. The loss measures how accurately the model predicts the noise added to normal time-series data. A lower loss means better denoising. Better denoising does not establish better anomaly detection. That needs a separate evaluation on labeled normal and anomalous data, for example using average precision. This recipe does not run that evaluation. The upstream guide shows how to run anomaly detection with the trained checkpoint (see Results below).

The validation loss is stochastic, so it changes from epoch to epoch. The epoch with the lowest validation loss becomes the best checkpoint. The workload takes about 3 minutes to run.

## Before you run

The workload downloads the upstream code, Python packages, model weights, and dataset when it starts, so it needs outbound access to GitHub, PyPI, Hugging Face, and Kaggle. The weights are Apache 2.0 and download without a Hugging Face token. The dataset downloads without a Kaggle account, but Kaggle lists its license as unknown, so review the terms on the dataset page before you use it. To write the checkpoints elsewhere, set `env_variables.OUTPUT_ROOT` to an existing Unity Catalog volume you can write to.

## Run

```bash
databricks air run -f workload.yaml
```

## Results

The AIR MLflow run has per-epoch `train_loss` and `val_loss`, plus `best_val_loss`. These files are copied to `<OUTPUT_ROOT>/<MLFLOW_RUN_ID>/` and uploaded to the run under `kumo_finetune/`:

- `checkpoints/best_finetuned_model.pth` and `checkpoints/final_finetuned_model.pth`: the best-validation and last-epoch checkpoints
- `checkpoints/finetune_config.yaml`: the model configuration needed to load a checkpoint
- `checkpoints/epoch_metrics.json`: the losses for each epoch
- `training_summary.json`: training parameters, the upstream commit, and the final and best losses

The MLflow run also keeps the prepared training and validation CSVs with their original row IDs and checksums (`kumo_finetune/data/`), plus the data-preparation and training logs. The [upstream fine-tuning guide](https://github.com/NVIDIA/Kumo-TS/tree/main/Kumo-Anomaly#fine-tuning) shows how to run anomaly detection with the best checkpoint and its config.
