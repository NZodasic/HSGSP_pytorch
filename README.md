# HSGSP: Hybrid Spectral-Guided Structured Pruning

This repository implements the HSGSP pipeline for training, pruning, and evaluating VGG-style image classifiers on CIFAR-10/100. The pipeline couples strong regularized training (mixup, cutout, frequency-aware regularizers) with a hybrid pruning baseline that blends DCT-based frequency analysis, gradient saliency, and iterative fine-tuning to produce compact models with bounded accuracy loss. Every experiment is reproducible from the command line and writes its artifacts under timestamped folders in `EXPERIMENT/`. This is the official implementation for our paper submitted to IEEE Latin America Transaction.



## Project layout

```
├── main.py                 # Orchestrates training, pruning, evaluation
├── config.py               # Dataclass with hyper-parameters + path builders
├── data/
│   └── data_loader.py      # CIFAR loaders with augmentation and mixup
├── models/
│   ├── vgg16.py            # Regularized VGG16 variant used throughout
│   └── model_utils.py      # FLOP/parameter counters and helpers
├── training/
│   ├── trainer.py          # HSGSPTrainer for baseline/fine-tune phases
│   ├── evaluator.py        # Rich evaluation metrics + benchmarking
│   ├── distillation.py     # Optional KD utilities
│   └── frequency_regularizer.py
├── pruning/
│   ├── hybrid_baseline.py  # Hybrid frequency + gradient pruning loop
│   ├── frequency_rel_analyzer.py  # Builds the FRN predictor
│   └── pruning_strategy.py # Structured channel pruning + regrow helpers
├── utils/                  # Logging, visualization, overfitting monitors
├── experiment.sh           # Convenience script to run the full pipeline
├── requirements.txt
└── EXPERIMENT/             # Auto-created run folders (logs, models, plots, …)
```

## Requirements

- Python 3.9–3.11 (TensorFlow 2.16.x support window).
- pip/virtualenv (or conda) to isolate dependencies.
- NVIDIA GPU with CUDA 12 **or** Apple Silicon GPU (TensorFlow-Metal is listed) recommended for reasonable runtimes. The code falls back to CPU if no accelerator is available.
- Internet access the first time you run each dataset so that `tf.keras.datasets` can download CIFAR-10/100.

Install dependencies into a fresh environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> If you need CUDA-enabled TensorFlow, make sure the matching CUDA/cuDNN runtimes are on your system before running `pip install`.

## Configuration

All defaults live in `config.py` (`Config` dataclass). When `Config` is instantiated it:

- sets data, training, pruning, FRN, and logging hyper-parameters,
- creates a unique `run_id` (timestamp) and corresponding folders under `EXPERIMENT/<run_id>_<task>/`.

Edit `config.py` if you want to change global defaults (e.g., optimizer, augmentation strength, pruning schedule, FRN architecture). Frequently tuned fields:

- **Data/training:** `task`, `batch_size`, `validation_split`, `default_epochs`, learning-rate schedule knobs.
- **Regularization:** dropout rates, label smoothing, mixup parameters.
- **Pruning:** `hybrid_iterations`, `hybrid_prune_fraction`, `hybrid_mode`, accuracy guard rails, min channel counts.
- **Fine-tuning:** `simple_finetune_*`, `hybrid_finetune_epochs`, `pruned_growth_lr`.
- **Frequency Relevance Net (FRN):** architecture, feature count, learning rate, entropy targets.

Many runtime overrides are exposed as CLI flags (see below) so you can keep `config.py` for “global defaults” and only override per-experiment settings from the command line.

## Running experiments

`main.py` is the single entry point. Combine flags to enable the phases you need:

```
python main.py \
  --task {cifar10|cifar100} \
  [--train] [--prune] [--simple_finetune] [--eval] \
  [--model_path path/to/baseline.h5] \
  [--pruned_model_path path/to/pruned.h5] \
  [--batch_size 256] [--epochs 150] [--gpu 0]
```

| Flag | Description |
| --- | --- |
| `--task` | Dataset to use. Controls class count and input shape. |
| `--train` | Train a fresh VGG16 baseline with the current config. |
| `--prune` | Run the hybrid pruning loop (FRN training → frequency + gradient scoring → structured channel pruning → fine-tuning). Expects either a freshly trained model from `--train` or a checkpoint passed via `--model_path`. |
| `--simple_finetune` | Run the light-weight fine-tune defined in `Config.simple_finetune_*` after pruning for extra recovery. |
| `--eval` | Evaluate whichever models are in memory. If both baseline and pruned variants exist, reports metrics for both. |
| `--model_path` | Path to an existing `.h5` baseline to skip training. |
| `--pruned_model_path` | Optionally preload a previously pruned model. |
| `--batch_size`, `--epochs` | Override defaults without editing `config.py`. |
| `--gpu` | Convenience flag so your launcher can set `CUDA_VISIBLE_DEVICES` (the script itself does not change device placement). |

### Common workflows

- **Full pipeline (train → prune → evaluate)**  
  ```bash
  python main.py --task cifar10 --train --prune --eval
  ```
  Saves the baseline to `EXPERIMENT/<run>/models/cifar10_trained_model.h5`, the pruned model to `.../cifar10_hybrid_pruned.h5`, plots to `.../plots/`, and logs metrics to `.../logs/*.log`.

- **Prune an existing checkpoint**  
  ```bash
  python main.py --task cifar100 --prune \
      --model_path path/to/cifar100_trained_model.h5
  ```
  Useful when you already have a trained network and want to iterate on pruning hyper-parameters.

- **Evaluation only**  
  ```bash
  python main.py --task cifar10 --eval \
      --model_path path/to/baseline.h5 \
      --pruned_model_path path/to/pruned.h5
  ```
  Produces accuracy, Top-5, inference latency, FLOPs, and parameter counts for each supplied model.

- **Simple manual fine-tune**  
  ```bash
  python main.py --task cifar10 --simple_finetune --pruned_model_path path/to/pruned.h5
  ```
  Runs the shorter fine-tuning schedule defined in `Config.simple_finetune_*` to squeeze out extra accuracy without executing the full hybrid loop again.

- **Batch script**  
  The included helper selects the dataset for you and runs the full loop:
  ```bash
  bash experiment.sh CIFAR-10
  bash experiment.sh CIFAR-100
  ```
  The argument must be the uppercase dataset name shown above.

## Outputs, logs, and monitoring

Each run creates `EXPERIMENT/<run_id>_<task>/` with:

- `tensorboard_logs/train` & `tensorboard_logs/val` – TensorBoard scalars written by `SplitTensorBoard`.
- `models/` – Saved `.h5` checkpoints (baseline, hybrid-pruned, finetuned variants).
- `logs/<timestamp>/HSGSP.log` – Console + file logger output.
- `results/` – Serialized metrics/artifacts written by the trainer/evaluator.
- `plots/` – PNGs such as the training history curve.

To launch TensorBoard:

```bash
tensorboard --logdir EXPERIMENT --port 6006
```

Evaluation writes confusion matrices, FLOPs, and latency measurements to the log and returns them to `main.py` for printing. Visual assets (frequency spectra, pruning histograms, compression comparisons) can be produced via the methods in `utils/visualization.py`.

## Tips & troubleshooting

- **Data download errors:** delete `~/.keras/datasets/cifar*` if the initial download was interrupted, then rerun.
- **GPU selection:** set `CUDA_VISIBLE_DEVICES=0` (or similar) before the `python main.py ...` call; the provided `--gpu` flag is just a convenience hook for wrappers.
- **Memory pressure:** lower `batch_size` with `--batch_size` or edit `Config.batch_size`, especially when running multiple pruning iterations.
- **Speeding up experiments:** reduce `Config.hybrid_iterations`, `hybrid_finetune_epochs`, or disable mixup/augmentation while debugging.
- **Custom datasets:** the repo is CIFAR-specific out of the box. To adapt it, extend `DataLoader` and update `_load_datasets` / `_build_model` in `main.py` plus the `Config` defaults for class counts and input shapes.

# Authors
- Thanh-Thien Nguyen - University of Information Technology, Vietnam National University Ho Chi Minh City
- Hoang-Loc Tran - University of Information Technology, Vietnam National University Ho Chi Minh City
- Vo-Chi-Dung Nguyen - University of Information Technology, Vietnam National University Ho Chi Minh City
- Viet-An Nguyen - University of Information Technology, Vietnam National University Ho Chi Minh City
- Duc-Lung Vu - University of Information Technology, Vietnam National University Ho Chi Minh City

