# Kumo-Anomaly fine-tuning for time-series anomaly detection

Fine-tune NVIDIA's [Kumo-Anomaly](https://huggingface.co/nvidia/Kumo-Anomaly) diffusion model on the public [Kaggle multivariate time-series dataset](https://www.kaggle.com/datasets/drscarlat/time-series) for 10 epochs on one A10 GPU. The recipe runs the fine-tuning example from NVIDIA's [Kumo-TS repository](https://github.com/NVIDIA/Kumo-TS/tree/main/Kumo-Anomaly) at a pinned commit, inside that project's locked `uv` environment.

Kumo-Anomaly learns what normal behavior looks like, so the recipe trains and validates on anomaly-free data only. The dataset has 11 numeric features and a 0/1 anomaly label for each row.

The recipe splits the series in time order into three parts: the first 60% for training, the next 20% for validation, and the last 20% held out as a test set. This recipe does not use that test set. It is reserved for measuring detection performance before and after fine-tuning (see [Evaluate on the held-out test set](#evaluate-on-the-held-out-test-set) below). Within the training and validation parts, the recipe scans the series in non-overlapping 100-row blocks and keeps only the blocks that contain no labeled anomalies. It keeps the earliest such blocks up to a fixed count: 500 blocks (50,000 rows) for training and 100 blocks (10,000 rows) for validation. Labels only decide which blocks to keep. The model never sees them.

The recipe reports the upstream training and validation losses. The model is trained to denoise normal data. It learns to predict the noise that was added to the data. The loss measures how well it does this. A lower loss means better denoising.

A low loss does not mean good anomaly detection. The two are different. To measure detection quality, you score labeled data that has both normal and anomalous rows, then compare the scores to the labels. A common metric is average precision. This recipe does not run that step. See [Evaluate on the held-out test set](#evaluate-on-the-held-out-test-set) below.

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

The MLflow run also keeps the prepared training and validation CSVs with their original row IDs and checksums (`kumo_finetune/data/`), plus the data-preparation and training logs. To score the held-out test set with a checkpoint, see [Evaluate on the held-out test set](#evaluate-on-the-held-out-test-set) below.

## Evaluate on the held-out test set

The recipe trains and validates on normal data, so its losses do not show how well the fine-tuned model detects anomalies. The last 20% of the series is held out for that. It contains both normal and anomalous rows with their labels, so it works as a labeled test set.

The steps below run outside the AIR workload, from the upstream `Kumo-Anomaly` directory and its `uv` environment, using the checkpoint the run produced (see [Results](#results)).

1. Build the test set. Read the source `TimeSeries.csv` (columns `v1`..`v11`) and `labelsTimeSeries.csv` (column `label`). Keep the rows after the 80% boundary, which `prepare_data.py` records as the third value of `temporal_boundaries` in `data/manifest.json`. Use the feature columns as the input and the labels for scoring.

2. Score the test set with the fine-tuned checkpoint. The upstream [`examples/quick_example.py`](https://github.com/NVIDIA/Kumo-TS/blob/main/Kumo-Anomaly/examples/quick_example.py) shows the detection call:

   ```python
   from sdk.anomaly_analysis import perform_anomaly_analysis_with_diffusion

   results = perform_anomaly_analysis_with_diffusion(
       df=features,  # the v1..v11 columns of the held-out rows
       threshold_strategy="scs",
       model_path="best_finetuned_model.pth",
       model_config_path="finetune_config.yaml",
       nsample=15,
   )
   ```

   `results` has one row per input row. `MAE` is the anomaly score and `Anomaly` is the 0/1 prediction.

3. Compare the scores to the labels. Average precision is a good fit because it uses the score directly and needs no threshold:

   ```python
   from sklearn.metrics import average_precision_score

   average_precision_score(labels, results["MAE"])
   ```

To see the effect of fine-tuning, score the same test set with the base `final_model.pth` and compare its average precision to the fine-tuned checkpoint's.
