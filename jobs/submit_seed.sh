#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -l gpu_card=1                 # GPU request (cluster-specific)
#$ -q gpu                        # GPU queue (cluster-specific) gpu@@crc_gpu
#$ -o logs/$JOB_NAME.o$JOB_ID
#$ -e logs/$JOB_NAME.e$JOB_ID
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p logs

# ===========================
#  ENVIRONMENT / MODULES
# ===========================
#CONDA_ENV="polygnn-gpu"          # Conda env with: torch(+cuda), torch-geometric, rdkit, optuna, numpy, pandas, matplotlib
module load cuda/11.8            # Adjust to your cluster
module load conda/24.7.1         # Adjust to your cluster
source ~/.bashrc
export MKL_INTERFACE_LAYER=""    # Avoid MKL conflicts on some clusters
#conda activate "${CONDA_ENV}"
ENV_ROOT="/users/salosiou/afs/.conda/envs/polygnn-gpu"
export PATH="$ENV_ROOT/bin:$PATH"
#source ~/.bashrc
conda activate polygnn-gpu
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
# ===========================
#  DATA / TARGETS / FIDELITIES
# ===========================
ROOT_DIR="data/raw"     # Data root. Discovery supports:
                                 #   1) ${ROOT_DIR}/{fid}/{target}.csv   e.g., exp/cp.csv
                                 #   2) ${ROOT_DIR}/${target}_${fid}.csv e.g., cp_exp.csv
                                 # Each CSV must have 'smiles' and <target> (or 'value'/'y').

#TARGETS="alpha,bandgap,dc,etotal,homo,pe,lumo,mu,ri,rg,rho,cp,tg,tm,td,tc,visc,dif,bulk,poisson,shear,young,phe,ph2,pco2,pn2,po2,pch4"  #all
#TARGETS="alpha,bandgap,dc,etotal,homo,pe,lumo,mu,ri" #electronic
#TARGETS="cp,tg,tm,td,tc" #thermal
#TARGETS="bulk,poisson,shear,young" #mechanical
#TARGETS="visc,dif,rho,rg" #other
#TARGETS="phe,ph2,pco2,pn2,po2,pch4" #permeability
#TARGETS="tc, dif, visc" #transport


#TARGETS="alpha, bandgap, homo, lumo, ri, etotal" #G1
#TARGETS="cp, lumo, bandgap, rg, rho, tg" #G2
#TARGETS="tg, tm, visc" #G3
#TARGETS="phe,ph2,pco2,pn2,po2,pch4,cp,tc" #G4

#TARGETS="cp, lumo" #G5
#TARGETS="cp, rho" #G6
#TARGETS="cp, tg" #G7
#TARGETS="cp, shear" #G8
#TARGETS="cp, pch4" #G9




# OPTIONAL: Only affects run naming & checkpoint SELECTION metric; does NOT affect loss weighting.
# Leave empty for overall metric selection.
SELECT_TASK=""                   # e.g., "rho" or "" to select by overall_<metric>

FIDELITIES="exp,dft,md,gc"           # Comma-separated fidelities to include (exp, dft, md, gc; case-insensitive).

# Base loss weights per fidelity (same order as FIDELITIES). Leave empty to use defaults (all equal, unless EXP schedule active).
FID_LOSS=""                     # e.g., "1.0,0.8,0.6"

# SMILES-level split file; created on first run and then reused.
SPLIT="splits/all.json"

# ===========================
#  TRAIN / HPO / METRIC
# ===========================
EPOCHS=120
BATCH_SIZE=64
N_TRIALS=100                      # 0 → no HPO; >0 → Optuna HPO trials
METRIC="rmse"                    # rmse | mae | r2

# ===========================
#  OPTIONAL KNOBS
# ===========================
# Multi-fidelity helpers
SAMPLER="balance_fidelity"       # "balance_fidelity" or "uniform"
EXP_SCHEDULE="none"              # EXP curriculum: none | linear | cosine
EXP_WEIGHT_START=0.6
EXP_WEIGHT_END=1.0
SELECT_FIDELITY="exp"            # If set, model selection uses <fid>_<metric>; must be one of FIDELITIES

# Multi-task loss balancing (DEFAULT: equal; leave off to keep equal importance)
TASK_UNCERTAINTY="1"              # "1" → enable homoscedastic task-uncertainty weighting; "" → equal importance

# Per-sample predictive uncertainty
HETEROSCEDASTIC="1"               # "1" → heteroscedastic heads (mean+logvar); "" → plain heads

# Train-only subsampling (learning curves)
SUBSAMPLE_TARGET=""              # e.g., "cp"
SUBSAMPLE_FIDELITY=""            # e.g., "gc"
SUBSAMPLE_PCT="1.0"              # 0 < pct ≤ 1.0; use 1.0 to disable
SUBSAMPLE_SEED="42"

# Results root; OUT_DIR and STUDY_NAME are auto-derived below
BASE_RESULTS_DIR="RESULTS_transport"

# ===========================
#  AUTO-NAMING (OUT_DIR / STUDY_NAME)
# ===========================
IFS=',' read -ra _T_ARR <<< "${TARGETS}"
NUM_T=${#_T_ARR[@]}

if (( NUM_T > 1 )); then
  # Multi-task naming
  if [[ -n "${SELECT_TASK}" ]]; then
    OUT_DIR="${BASE_RESULTS_DIR}/multitask_select_${SELECT_TASK}"
    STUDY_NAME="multitask_${SELECT_TASK}_hpo"
  else
    OUT_DIR="${BASE_RESULTS_DIR}/multitask_overall"
    STUDY_NAME="multitask_overall_hpo"
  fi
else
  # Single-task naming
  SINGLE_TASK="${_T_ARR[0]}"
  OUT_DIR="${BASE_RESULTS_DIR}/single_${SINGLE_TASK}"
  STUDY_NAME="single_${SINGLE_TASK}_hpo"
fi
mkdir -p "${OUT_DIR}"

# ===========================
#  FIDELITY COUNT & SINGLE-FID GUARD
# ===========================
IFS=',' read -ra _F_ARR <<< "${FIDELITIES}"
NUM_F=${#_F_ARR[@]}

# Single fidelity: disable curriculum & balancing; selection by fidelity is meaningless.
if [[ ${NUM_F} -eq 1 ]]; then
  FID_LOSS="1.0"
  EXP_SCHEDULE="none"
  SAMPLER="uniform"
  SELECT_FIDELITY=""
fi

# ===========================
#  MULTI-SEED
# ===========================
#SEEDS=()   # Set empty () to run a single seed
SEEDS=("42" "137" "2023" "7" "99")   # Set empty () to run a single seed

# ===========================
#  BUILD ARG LIST
# ===========================
args=()
args+=(--root_dir "${ROOT_DIR}")
args+=(--targets "${TARGETS}")
args+=(--fidelities "${FIDELITIES}")
args+=(--split_file "${SPLIT}")
args+=(--out_dir "${OUT_DIR}")
args+=(--epochs "${EPOCHS}")
args+=(--batch_size "${BATCH_SIZE}")
args+=(--device cuda)
args+=(--n_trials "${N_TRIALS}")
args+=(--study_name "${STUDY_NAME}")
args+=(--pruner median)
args+=(--metric "${METRIC}")

# ---- Multi-fidelity options ----
if [[ ${NUM_F} -gt 1 ]]; then
  [[ -n "${SAMPLER}" ]] && args+=(--sampler "${SAMPLER}")
  if [[ -n "${EXP_SCHEDULE}" && "${EXP_SCHEDULE}" != "none" ]]; then
    args+=(--exp_weight_schedule "${EXP_SCHEDULE}" --exp_weight_start "${EXP_WEIGHT_START}" --exp_weight_end "${EXP_WEIGHT_END}")
  fi
  [[ -n "${SELECT_FIDELITY}" ]] && args+=(--select_fidelity "${SELECT_FIDELITY}")
fi

# Always pass fid loss if explicitly set
[[ -n "${FID_LOSS}" ]] && args+=(--fid_loss_w "${FID_LOSS}")

# ---- Multi-task selection / options (loss stays equal unless you toggle TASK_UNCERTAINTY) ----
if [[ ${NUM_T} -gt 1 ]]; then
  [[ -n "${TASK_UNCERTAINTY}" ]] && args+=(--task_uncertainty)   # optional deviation from equal-importance
  [[ -n "${SELECT_TASK}" ]] && args+=(--select_task "${SELECT_TASK}")
fi

# ---- Heteroscedastic prediction head ----
[[ -n "${HETEROSCEDASTIC}" ]] && args+=(--heteroscedastic)

# ---- Train-only subsampling (learning curves) ----
if [[ -n "${SUBSAMPLE_TARGET}" && -n "${SUBSAMPLE_FIDELITY}" && "${SUBSAMPLE_PCT}" != "1.0" ]]; then
  args+=(--subsample_target "${SUBSAMPLE_TARGET}" --subsample_fidelity "${SUBSAMPLE_FIDELITY}" --subsample_pct "${SUBSAMPLE_PCT}" --subsample_seed "${SUBSAMPLE_SEED}")
fi

# ===========================
#  RUN (seed loop)
# ===========================
if (( ${#SEEDS[@]} == 0 )); then
  python scripts/train.py "${args[@]}"
else
  for SEED in "${SEEDS[@]}"; do
    SEED_OUT="${OUT_DIR}/seed_${SEED}"
    mkdir -p "${SEED_OUT}"
    python scripts/train.py "${args[@]}" --out_dir "${SEED_OUT}" --seed "${SEED}"
  done
fi

# ===========================
#  OPTIONAL POST-RUN (requires your own helper scripts)
# ===========================
if [[ -f "plots/aggregate_seeds.py" ]]; then
  python plots/aggregate_seeds.py --runs_root "${OUT_DIR}" --out_name "seed_summary"
  for TASK in $(echo "${TARGETS}" | tr ',' ' '); do
    [[ -f "plots/calibration_across_seeds.py" ]] && python plots/calibration_across_seeds.py --runs_root "${OUT_DIR}" --task "${TASK}" --split "test"
    [[ -f "plots/median_seed_plots.py" ]] && python plots/median_seed_plots.py --runs_root "${OUT_DIR}" --task "${TASK}" --split "test"
    [[ -f "plots/ensemble_from_seeds.py" ]] && python plots/ensemble_from_seeds.py --runs_root "${OUT_DIR}" --task "${TASK}" --split "test"
  done
else
  echo "[info] Optional post-run helpers not found under plots/. Skipping aggregation hooks."
fi

# ===========================
#  NOTES
# ===========================
# - Equal importance across tasks is default (no --task_weights, no --task_uncertainty).
# - To focus model SELECTION on a property without changing training loss, set SELECT_TASK (keeps training equal).
# - Keep the same --split_file across ST/MT for apples-to-apples comparison.
