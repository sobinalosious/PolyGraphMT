#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -l gpu_card=1                 # GPU request (cluster-specific)
#$ -q gpu@@crc_gpu                       # GPU queue (cluster-specific)
#$ -o logs/$JOB_NAME.o$JOB_ID.$TASK_ID
#$ -e logs/$JOB_NAME.e$JOB_ID.$TASK_ID
#$ -t 1-3                       # 28 properties → 28 array tasks
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

# List of ALL properties; each array task picks ONE of these
PROPERTIES=(td visc dif )

#PROPERTIES=(
#  alpha bandgap dc etotal homo pe lumo mu ri rg rho cp tg tm td tc
#  visc dif bulk poisson shear young phe ph2 pco2 pn2 po2 pch4)

NUM_PROPERTIES=${#PROPERTIES[@]}
TASK_ID=${SGE_TASK_ID:-1}
IDX=$((TASK_ID - 1))

if (( IDX < 0 || IDX >= NUM_PROPERTIES )); then
  echo "ERROR: SGE_TASK_ID=${TASK_ID} is out of range 1-${NUM_PROPERTIES}"
  exit 1
fi

SINGLE_TASK="${PROPERTIES[$IDX]}"
echo "==============================="
echo " SGE_TASK_ID = ${TASK_ID}"
echo " SINGLE_TASK = ${SINGLE_TASK}"
echo "==============================="

# OPTIONAL: For single-task runs this can stay empty
SELECT_TASK=""

FIDELITIES="exp,dft,md,gc"      # Comma-separated fidelities to include (exp, dft, md, gc; case-insensitive).

# Base loss weights per fidelity (same order as FIDELITIES). Leave empty to use defaults (all equal, unless EXP schedule active).
FID_LOSS=""                     # e.g., "1.0,0.8,0.6"

# SMILES-level split file; created on first run and then reused.
SPLIT="splits/all.json"

# ===========================
#  TRAIN / HPO / METRIC
# ===========================
EPOCHS=120
BATCH_SIZE=64
N_TRIALS=100                    # 0 → no HPO; >0 → Optuna HPO trials
METRIC="rmse"                   # rmse | mae | r2

# ===========================
#  OPTIONAL KNOBS
# ===========================
# Multi-fidelity helpers
SAMPLER="balance_fidelity"      # "balance_fidelity" or "uniform"
EXP_SCHEDULE="none"             # EXP curriculum: none | linear | cosine
EXP_WEIGHT_START=0.6
EXP_WEIGHT_END=1.0
SELECT_FIDELITY="exp"           # If set, model selection uses <fid>_<metric>; must be one of FIDELITIES

# Multi-task loss balancing (not used here; single-task only)
TASK_UNCERTAINTY="1"

# Per-sample predictive uncertainty
HETEROSCEDASTIC="1"              # "1" → heteroscedastic heads (mean+logvar); "" → plain heads

# Train-only subsampling (learning curves)
SUBSAMPLE_TARGET=""             # e.g., "cp"
SUBSAMPLE_FIDELITY=""           # e.g., "gc"
SUBSAMPLE_PCT="1"             # 0 < pct ≤ 1.0; use 1.0 to disable
SUBSAMPLE_SEED="42"

# Results root: we now want RESULTS_Single/{property}
BASE_RESULTS_ROOT="RESULTS_Single"
OUT_DIR="${BASE_RESULTS_ROOT}/${SINGLE_TASK}"
STUDY_NAME="single_${SINGLE_TASK}_hpo"
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
SEEDS=("42" "137" "2023" "7" "99")   # Set empty () to run a single seed

# ===========================
#  BUILD ARG LIST FOR THIS PROPERTY
# ===========================
args=()
args+=(--root_dir "${ROOT_DIR}")
args+=(--targets "${SINGLE_TASK}")        # SINGLE-TASK only
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

# ---- Heteroscedastic prediction head ----
[[ -n "${HETEROSCEDASTIC}" ]] && args+=(--heteroscedastic)

# ---- Train-only subsampling (learning curves) ----
if [[ -n "${SUBSAMPLE_TARGET}" && -n "${SUBSAMPLE_FIDELITY}" && "${SUBSAMPLE_PCT}" != "1.0" ]]; then
  args+=(--subsample_target "${SUBSAMPLE_TARGET}" --subsample_fidelity "${SUBSAMPLE_FIDELITY}" --subsample_pct "${SUBSAMPLE_PCT}" --subsample_seed "${SUBSAMPLE_SEED}")
fi

# ===========================
#  RUN (seed loop) FOR THIS PROPERTY
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
#  POST-RUN (per property)
# ===========================
if [[ -f "plots/aggregate_seeds.py" ]]; then
  python plots/aggregate_seeds.py --runs_root "${OUT_DIR}" --out_name "seed_summary"
  [[ -f "plots/calibration_across_seeds.py" ]] && python plots/calibration_across_seeds.py --runs_root "${OUT_DIR}" --task "${SINGLE_TASK}" --split "test"
  [[ -f "plots/median_seed_plots.py" ]] && python plots/median_seed_plots.py --runs_root "${OUT_DIR}" --task "${SINGLE_TASK}" --split "test"
  [[ -f "plots/ensemble_from_seeds.py" ]] && python plots/ensemble_from_seeds.py --runs_root "${OUT_DIR}" --task "${SINGLE_TASK}" --split "test"
else
  echo "[info] Optional post-run helpers not found under plots/. Skipping aggregation hooks."
fi

# ===========================
#  NOTES
# ===========================
# - This is an SGE job array: each TASK_ID (1..28) trains ONE property.
# - Output structure: RESULTS_Single/{property}/seed_{SEED}/...
# - If you add/remove properties in PROPERTIES, update the -t range accordingly.
