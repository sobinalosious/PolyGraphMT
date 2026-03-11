# PolyGraphMT

PolyGraphMT is a graph neural network workflow for polymer property prediction with:

- multiple target properties trained together
- multiple fidelity levels per property
- SMILES-based molecular graph construction with RDKit
- PyTorch Geometric backbones (`gine`, `gin`, `gcn`)
- optional fidelity conditioning, task embeddings, uncertainty heads, and Optuna tuning

The repository is organized so it can be installed as a standard Python project, run locally from scripts, or submitted through cluster job scripts.

## What The Repository Does

At a high level, the workflow is:

1. Read polymer-property CSV files from `data/raw/`
2. Build molecular graphs from SMILES strings with RDKit
3. Merge all requested targets and fidelities into one training table
4. Split by unique SMILES so the same polymer does not leak across train/val/test
5. Normalize targets per task, with automatic `log10` transforms when appropriate
6. Train a multi-task, multi-fidelity GNN
7. Save checkpoints, metrics, prediction CSVs, and parity/calibration plots
8. Reuse the saved checkpoint and scaler for inference on new SMILES

## Repository Layout

```text
.
├── models/                   # curated released checkpoints and scaler files
├── data/
│   ├── README.md
│   └── raw/                  # input CSV files
├── jobs/                     # SGE cluster submission scripts
├── scripts/                  # lightweight CLI wrappers
├── src/
│   └── polygraphmt/          # installable Python package
├── pyproject.toml
└── README.md
```

Key code files:

- `src/polygraphmt/train.py`: training, evaluation, Optuna HPO
- `src/polygraphmt/predict.py`: inference on new SMILES
- `src/polygraphmt/data_builder.py`: CSV discovery, graph featurization, splits, scaling
- `src/polygraphmt/model.py`: multi-task multi-fidelity model
- `src/polygraphmt/conv.py`: GNN encoder blocks
- `src/polygraphmt/plot_utils.py`: parity and calibration plots from train/val/test predictions
- `src/polygraphmt/results_summary.py`: summarize many `RESULTS_*` folders into CSVs and bar plots

## Requirements

This project depends on standard Python packages plus a few heavier ML and chemistry packages.

Core dependencies already listed in `pyproject.toml`:

- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `tqdm`
- `optuna`

Additional required packages:

- `torch`
- `torch-geometric`
- `rdkit`

`torch`, `torch-geometric`, and `rdkit` are environment-sensitive and should be installed in variants that match the target system and CUDA version.

## Installation

The repository includes two environment definitions:

- [environment.yml](environment.yml): Conda environment definition
- [requirements.txt](requirements.txt): pip requirements file

### Option A: Conda environment

```bash
conda env create -f environment.yml
conda activate polygraphmt
```

### Option B: pip requirements

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Environment notes:

- Conda is typically the most stable installation path for `rdkit`
- `requirements.txt` is intended for local and CPU-oriented workflows
- CUDA-specific PyTorch builds require installing the appropriate `torch` package before installing the remaining dependencies

Verify the CLI entry points:

```bash
python3 scripts/train.py --help
python3 scripts/predict.py --help
python3 scripts/predict_from_models.py --help
python3 scripts/summarize_results.py --help
```

## After Cloning This Repository

To run the code with the same dataset files included in the repository:

1. Enter the repository directory.

```bash
cd /path/to/PolyGraphMT
```

2. Create and activate the environment.

Conda:

```bash
conda env create -f environment.yml
conda activate polygraphmt
```

pip:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Run training using the dataset already included in `data/raw/`.

```bash
python3 scripts/train.py \
  --root_dir data/raw \
  --targets cp,tg,tc \
  --fidelities exp,dft,md,gc \
  --split_file splits/cp_tg_tc.json \
  --out_dir RESULTS_example/multitask_overall \
  --epochs 100 \
  --batch_size 64 \
  --device cpu
```

On systems with a CUDA-enabled GPU, replace `--device cpu` with `--device cuda`.

For comparable reruns, keep the following fixed:

- `--root_dir`
- `--targets`
- `--fidelities`
- `--split_file`
- `--seed`
- all model and optimizer settings

The training output directory contains the saved checkpoint, scaler, metrics JSON files, prediction CSV files, and generated plots.

## Released Model Artifacts

The repository also includes curated checkpoints under `models/`.

- `models/single_models/` contains released single-task checkpoints
- `models/multitask_models/` contains released multitask checkpoints
- `models/best_model_map.json` defines which model family should be used for each property

Only one seed is retained for each released model group in order to keep the repository smaller and easier to distribute.

For artifact-based inference:

- use `python3 scripts/predict_from_models.py ...` to resolve the correct released checkpoint automatically
- use `python3 scripts/predict.py --ckpt_path ... --scaler_path ...` when selecting a specific artifact pair manually

Additional details are documented in `models/README.md`.

## Input Data Format

### Supported directory layouts

The loader supports either of these layouts under `data/raw/`:

1. Folder layout

```text
data/raw/
├── exp/
│   ├── cp.csv
│   └── tg.csv
├── md/
│   ├── cp.csv
│   └── tg.csv
```

2. Flat filename layout

```text
data/raw/
├── CP_EXP.csv
├── TG_EXP.csv
├── CP_MD.csv
└── TG_MD.csv
```

Matching is case-insensitive.

### Required CSV columns

Each CSV must contain:

- `smiles`
- one value column for that target

The code accepts the target column directly, and also supports common alternatives when reading:

- the target name itself, for example `cp`
- `value`
- `y`

If duplicate SMILES appear within one CSV, the loader averages them.

### Fidelities

The internal default fidelity priority is:

`exp, dft, md, gc`

Any subset and any order may be used through `--fidelities`, but the order should remain consistent across comparative runs.

## Reproducibility Rules

For reproducible and comparable experiments, keep the following fixed:

- use the same `--split_file` so train/val/test SMILES stay identical
- keep `--seed` fixed
- keep the same target list and fidelity list
- keep the same metric selection rule (`--metric`, `--select_task`, `--select_fidelity`)
- keep the same data files in `data/raw/`

The split is performed by unique SMILES, not by row. This prevents the same polymer from appearing in multiple splits under different fidelities.

## Quick Start

### 1. Multi-task training

```bash
python3 scripts/train.py \
  --root_dir data/raw \
  --targets cp,tg,tc \
  --fidelities exp,dft,md,gc \
  --split_file splits/cp_tg_tc.json \
  --out_dir RESULTS_example/multitask_overall \
  --epochs 100 \
  --batch_size 64 \
  --device cuda
```

### 2. Single-task training

```bash
python3 scripts/train.py \
  --root_dir data/raw \
  --targets cp \
  --fidelities exp,dft,md,gc \
  --split_file splits/cp.json \
  --out_dir RESULTS_single_cp \
  --epochs 100 \
  --batch_size 64 \
  --device cuda
```

### 3. Train with Optuna HPO

```bash
python3 scripts/train.py \
  --root_dir data/raw \
  --targets cp,tg,tc \
  --fidelities exp,dft,md,gc \
  --split_file splits/cp_tg_tc.json \
  --out_dir RESULTS_hpo \
  --n_trials 50 \
  --study_name cp_tg_tc_hpo \
  --epochs 100
```

When `--n_trials > 0`, Optuna searches for the best configuration and then retrains it in:

`<out_dir>/best_run`

### 4. Predict on new SMILES

From a CSV:

```bash
python3 scripts/predict.py \
  --run_dir RESULTS_hpo/best_run \
  --input_csv my_smiles.csv \
  --fidelity exp \
  --out_csv predictions.csv
```

From a comma-separated SMILES string:

```bash
python3 scripts/predict.py \
  --run_dir RESULTS_hpo/best_run \
  --smiles "CCO,c1ccccc1" \
  --fidelity exp \
  --out_csv predictions.csv
```

Direct artifact mode is also supported:

```bash
python3 scripts/predict.py \
  --ckpt_path models/single_models/tg_single_model_42.pt \
  --scaler_path models/single_models/tg_single_scalar_42.pt \
  --input_csv my_smiles.csv \
  --fidelity exp \
  --out_csv predictions_tg.csv
```

For released checkpoints under `models/`, the helper wrapper resolves the correct artifact pair automatically:

```bash
python3 scripts/predict_from_models.py \
  --property cp \
  --input_csv my_smiles.csv \
  --fidelity exp \
  --out_csv predictions_cp.csv
```

### 5. Summarize many results folders

```bash
python3 scripts/summarize_results.py \
  --root . \
  --out_dir RESULTS_SUMMARY
```

## Training Outputs

Each completed training run writes a folder like:

```text
RESULTS_example/
└── best_run_or_plain_run/
    ├── best.pt
    ├── target_scaler.pt
    ├── run_config.json
    ├── history.json
    ├── metrics_train.json
    ├── metrics_val.json
    ├── metrics_test.json
    ├── predictions_train.csv
    ├── predictions_val.csv
    ├── predictions_test.csv
    ├── parity_val_<task>.png
    ├── parity_test_<task>.png
    ├── parity_all_<task>.png
    └── calibration_<split>_<task>.png   # only when heteroscedastic output is enabled
```

If Optuna is used, the parent output directory also contains:

- `optuna_results.json`

### What the main files mean

- `best.pt`: best model checkpoint selected by validation metric
- `target_scaler.pt`: saved target normalization and inverse-transform metadata
- `run_config.json`: exact arguments used for the run
- `history.json`: epoch-by-epoch training loss and validation metrics
- `metrics_*.json`: summary metrics for each split
- `predictions_*.csv`: per-sample predictions in final reported units

Fixed post-scaling is applied automatically after inverse scaling for:

- `td`: multiply by `1e-7`
- `dif`: multiply by `1e-5`
- `visc`: multiply by `1e-3`

This correction is applied in the current training/evaluation and prediction code, so exported predictions, uncertainties, and `_orig` metrics use the final reported units.

The metrics JSON files contain both:

- normalized or transformed-space metrics such as `overall_rmse`
- reported-unit metrics such as `overall_rmse_orig`

### Prediction CSV columns

For each split, the saved CSV contains:

- `split`
- `smiles`
- `fid`
- `fid_idx`
- `y_<task>`
- `pred_<task>`
- `mask_<task>`

When heteroscedastic uncertainty is enabled, it also includes:

- `std_<task>`
- `pi95_lo_<task>`
- `pi95_hi_<task>`

## Training Arguments

Below is the practical meaning of the main settings in `scripts/train.py`.

### Data arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `--root_dir` | required | Root directory containing target/fidelity CSV files |
| `--targets` | required | Comma-separated list of target properties to train |
| `--fidelities` | `exp,dft,md,gc` | Fidelities to include |
| `--val_ratio` | `0.15` | Fraction of unique SMILES used for validation |
| `--test_ratio` | `0.15` | Fraction of unique SMILES used for testing |
| `--split_file` | `None` | JSON file used to save or reuse a fixed SMILES split |

For controlled comparisons:

- set `--split_file` for every serious experiment
- use the same split file when comparing models

### Model backbone arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `--gnn_type` | `gine` | Graph convolution type: `gine`, `gin`, or `gcn` |
| `--gnn_emb_dim` | `256` | Hidden embedding size in the GNN encoder |
| `--gnn_layers` | `5` | Number of graph message-passing layers |
| `--gnn_norm` | `batch` | Normalization layer inside the encoder |
| `--gnn_readout` | `mean` | Graph pooling method |
| `--gnn_act` | `relu` | Encoder activation function |
| `--gnn_dropout` | `0.0` | Dropout inside the encoder |
| `--no_gnn_residual` | off | Disable residual connections |

Typical use:

- `gine` is the chemistry-oriented default because it uses edge attributes
- larger `gnn_emb_dim` and more `gnn_layers` increase capacity and cost
- `layer` norm can sometimes be more stable than `batch` norm for smaller batches

### Fidelity and task conditioning arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `--fid_emb_dim` | `64` | Fidelity embedding size |
| `--no_film` | off | Disable FiLM-based fidelity conditioning |
| `--no_task_embed` | off | Disable learned task embeddings |
| `--task_emb_dim` | `32` | Task embedding size |

Typical use:

- FiLM is typically left enabled unless the experiment explicitly tests the no-conditioning case
- task embeddings are usually helpful for multi-task learning

### Prediction head and uncertainty arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `--head_hidden` | `512` | Hidden width of each per-task head |
| `--head_depth` | `2` | Number of hidden layers in each head |
| `--head_act` | `relu` | Head activation |
| `--head_dropout` | `0.0` | Head dropout |
| `--heteroscedastic` | off | Predict per-sample uncertainty as mean + log-variance |
| `--task_uncertainty` | off | Learn per-task homoscedastic uncertainty weights in the loss |

Distinction:

- `--heteroscedastic` changes the model output and gives per-sample uncertainty
- `--task_uncertainty` changes loss weighting across tasks during training

### Optimization arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `--epochs` | `100` | Number of training epochs |
| `--batch_size` | `64` | Batch size |
| `--lr` | `3e-4` | Learning rate |
| `--weight_decay` | `1e-5` | AdamW weight decay |
| `--grad_clip` | `1.0` | Gradient clipping threshold |
| `--no_amp` | off | Disable automatic mixed precision |
| `--seed` | `42` | Random seed |
| `--device` | `cuda` | `cuda` or `cpu` |

### Loss weighting and model-selection arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `--task_weights` | `None` | Manual per-task loss weights |
| `--fid_loss_w` | `None` | Manual per-fidelity loss weights |
| `--exp_weight_schedule` | `none` | Schedule for EXP fidelity weight: `none`, `linear`, `cosine` |
| `--exp_weight_start` | `0.6` | Starting EXP weight for curriculum |
| `--exp_weight_end` | `1.0` | Final EXP weight for curriculum |
| `--metric` | `rmse` | Validation metric for model selection |
| `--select_task` | `None` | Select best checkpoint using `<task>_<metric>` |
| `--select_fidelity` | `None` | Select best checkpoint using `<fid>_<metric>` |
| `--sampler` | `uniform` | Train sampler: `uniform` or `balance_fidelity` |

Selection and weighting behavior:

- if `--select_task` is set, validation selection uses that task metric first
- if `--select_fidelity` is set, validation selection uses that fidelity metric next
- otherwise the code falls back to `overall_<metric>`
- `balance_fidelity` oversamples rare fidelities during training
- if an EXP schedule is used without explicit fidelity weights, non-EXP fidelities default to `0.4`

Checkpoint selection:

- checkpoint selection is based on normalized or transformed-space metrics such as `overall_rmse`
- reported-unit metrics with `_orig` are still computed and saved for interpretation and reporting

### Embedding regularization arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `--fid_emb_l2` | `0.0` | L2 penalty on fidelity embeddings |
| `--task_emb_l2` | `0.0` | L2 penalty on task embeddings |

These settings control explicit L2 penalties on the learned fidelity and task embeddings.

### Hyperparameter optimization arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `--n_trials` | `0` | Number of Optuna trials; `0` disables HPO |
| `--pruner` | `median` | Optuna pruner: `none` or `median` |
| `--study_name` | `None` | Optional Optuna study name |

When HPO is enabled, the code currently searches over:

- `gnn_emb_dim`
- `gnn_layers`
- `gnn_dropout`
- `head_hidden`
- `head_depth`
- `use_film`
- `fid_emb_dim` when FiLM is enabled
- `gnn_norm`
- `gnn_act`
- `gnn_type`
- `lr`
- `weight_decay`

During HPO, some settings are intentionally fixed:

- `gnn_readout = mean`
- `gnn_residual = True`
- `use_task_embed = True`
- `task_emb_dim = 32`
- `head_act = relu`
- `head_dropout = 0.0`
- `heteroscedastic = False`
- EXP curriculum is forced to cosine from `0.6` to `1.0`

### Train-only subsampling arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `--subsample_target` | `None` | Target to subsample only within the training set |
| `--subsample_fidelity` | `None` | Fidelity block to subsample |
| `--subsample_pct` | `1.0` | Fraction of that block to keep |
| `--subsample_seed` | `137` | Seed for deterministic subsampling |

This setting supports learning-curve style experiments, for example training with only 10% of `cp` at `gc` fidelity while leaving the rest of the dataset unchanged.

## Inference Arguments

`scripts/predict.py` accepts:

| Argument | Default | Meaning |
| --- | --- | --- |
| `--run_dir` | required | Directory containing `best.pt` and `target_scaler.pt` |
| `--input_csv` | `None` | CSV containing a `smiles` column |
| `--smiles` | `None` | Comma-separated SMILES string |
| `--fidelity` | `None` | Fidelity name or fidelity index |
| `--batch_size` | `256` | Inference batch size |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--out_csv` | `None` | Output CSV path; default is inside `run_dir` |

Inference rules:

- exactly one of `--input_csv` or `--smiles` is required
- single-fidelity checkpoints default to `exp`
- inference fails if the requested fidelity is absent from the checkpoint metadata

The output CSV contains:

- `smiles`
- `fid`
- `pred_<task>`

## Plotting Scripts

### Multi-task histogram plotting

```bash
python3 scripts/plot_hist_mt.py \
  --csv RESULTS_G4/multitask_overall/ensemble_predictions_POLYINFO.csv \
  --out_dir results_hist
```

Arguments:

| Argument | Default | Meaning |
| --- | --- | --- |
| `--csv` | required | Input ensemble CSV containing `mean_*` columns |
| `--out_dir` | required | Output directory |
| `--prop` | `None` | Plot only one property |
| `--bins` | `100` | Number of histogram bins |
| `--title_prefix` | `""` | Optional title prefix |
| `--stats_csv` | `None` | Optional path for summary stats CSV |

### Single-task histogram plotting

```bash
python3 scripts/plot_hist_st.py \
  --csv RESULTS_Single/cp/ensemble_cp_POLYINFO.csv \
  --prop cp \
  --out_dir results_hist_st/POLYINFO/cp
```

Arguments:

| Argument | Default | Meaning |
| --- | --- | --- |
| `--csv` | required | Single-task ensemble CSV |
| `--prop` | required | Property key |
| `--out_dir` | required | Output directory |
| `--bins` | `100` | Number of bins |
| `--title` | `None` | Optional title |
| `--stats_csv` | `None` | Optional path for summary stats CSV |

## Result Summary Script

This script summarizes several `RESULTS_*` directories into CSV tables and bar charts.

```bash
python3 scripts/summarize_results.py \
  --root . \
  --folder_prefix RESULTS_ \
  --out_dir RESULTS_SUMMARY
```

Arguments:

| Argument | Default | Meaning |
| --- | --- | --- |
| `--root` | `.` | Directory to scan |
| `--folder_prefix` | `RESULTS_` | Folder prefix to include |
| `--out_dir` | `RESULTS_SUMMARY` | Output folder |

## Automatic Target Transforms

The training pipeline automatically decides whether each target should use:

- `identity`
- `log10`

This decision is based on the training split only. A target is transformed with `log10` when it is mostly positive and spans a large dynamic range.

This behavior is implemented in `src/polygraphmt/data_builder.py`.

Implementation details:

- automatic transforms are enabled internally by default
- there is not yet a CLI flag to turn this on or off from `scripts/train.py`
- changing that behavior currently requires editing the code

Published results should state whether the default auto-transform logic was used unchanged.

## Cluster Job Scripts

The `jobs/` directory contains SGE-oriented scripts:

- `jobs/submit_seed.sh`: multi-seed training
- `jobs/submit_seed_single.sh`: array-based single-task training
- `jobs/submit_pred.sh`: batch prediction plus histogram workflow

These scripts are templates, not universal launchers. Before using them, update:

- module loads
- conda environment path
- queue names
- GPU resource requests
- input file names
- target groups

The scripts now assume the repo root layout introduced here:

- training entry point: `scripts/train.py`
- prediction entry point: `scripts/predict.py`
- input data root: `data/raw`

Some post-processing helpers referenced from `plots/` or `aggregate_seed_predictions.py` are not part of this repository. The job scripts skip those steps when the helper files are absent.

## Common Experiment Patterns

### Standard multi-task baseline

Common baseline settings:

- `--gnn_type gine`
- `--sampler balance_fidelity` when fidelities are imbalanced
- `--split_file ...`
- `--metric rmse`

Example:

```bash
python3 scripts/train.py \
  --root_dir data/raw \
  --targets cp,tg,tc,rho \
  --fidelities exp,dft,md,gc \
  --split_file splits/baseline.json \
  --out_dir RESULTS_baseline \
  --sampler balance_fidelity \
  --epochs 120
```

### Focus checkpoint selection on one task

This changes checkpoint selection only unless `--task_weights` or `--task_uncertainty` is also set.

```bash
python3 scripts/train.py \
  --root_dir data/raw \
  --targets cp,tg,tc,rho \
  --fidelities exp,dft,md,gc \
  --split_file splits/focus_rho.json \
  --out_dir RESULTS_focus_rho \
  --select_task rho
```

### Focus on one fidelity for validation selection

```bash
python3 scripts/train.py \
  --root_dir data/raw \
  --targets cp,tg,tc \
  --fidelities exp,dft,md,gc \
  --split_file splits/focus_exp.json \
  --out_dir RESULTS_focus_exp \
  --select_fidelity exp
```

### Add predictive uncertainty

```bash
python3 scripts/train.py \
  --root_dir data/raw \
  --targets cp,tg,tc \
  --fidelities exp,dft,md,gc \
  --split_file splits/uncertainty.json \
  --out_dir RESULTS_uncertainty \
  --heteroscedastic
```

### Learning-curve style subsampling

```bash
python3 scripts/train.py \
  --root_dir data/raw \
  --targets cp,tg,tc \
  --fidelities exp,dft,md,gc \
  --split_file splits/subsample.json \
  --out_dir RESULTS_subsample \
  --subsample_target cp \
  --subsample_fidelity gc \
  --subsample_pct 0.1 \
  --subsample_seed 42
```

## Troubleshooting

### `RDKit failed to parse SMILES`

The offending SMILES is skipped. Clean the input data if too many samples are being dropped.

### `Missing checkpoint: .../best.pt`

Inference expects a completed training run directory containing:

- `best.pt`
- `target_scaler.pt`

### `--fidelity ... not in trained fidelities`

The inference fidelity must match the fidelity set used when the model was trained.

### `python: command not found`

Use `python3` on systems where `python` is not available on `PATH`.

## Reproducing A Published Run

To reproduce one exact run, keep:

- identical code
- identical environment
- identical CSV inputs
- identical `run_config.json`
- identical `split_file`
- identical random seed

Reproduction procedure:

1. start from the same commit
2. recreate the environment
3. use the same input data under `data/raw/`
4. copy the command from `run_config.json`
5. rerun with the same `--split_file` and `--seed`

## Additional Notes

- generated results, caches, split files, and similar artifacts are ignored by `.gitignore`
- the included CSV files under `data/raw/` serve as in-repo inputs for the current project layout
- environment definitions are provided in `requirements.txt` and `environment.yml`
