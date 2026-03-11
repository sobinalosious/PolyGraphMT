#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -N mt_pred_agg_hist
#$ -q gpu
#$ -l gpu_card=1
#$ -t 1-9
#$ -o logs/$JOB_NAME.o$JOB_ID.$TASK_ID
#$ -e logs/$JOB_NAME.e$JOB_ID.$TASK_ID
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p logs

# ===========================
#  ENVIRONMENT (CRC-safe)
# ===========================
ENV_PREFIX="/afs/crc.nd.edu/user/s/salosiou/.conda/envs/polygnn-gpu"

module load cuda/11.8
module load conda/24.7.1
source ~/.bashrc

export MKL_INTERFACE_LAYER=""
conda activate "$ENV_PREFIX"
module purge || true

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"

PYBIN="$ENV_PREFIX/bin/python"

# ===========================
#  SETTINGS (EDIT HERE)
# ===========================
SMILES_SOURCE="PI1M"                 # POLYINFO | PI1M
INPUT_CSV="${SMILES_SOURCE}_SMILES.csv"

FIDELITY="exp"                           # exp | gc | md | dft
DEVICE="cuda"
BATCH_SIZE=256
CHUNKSIZE=200000
SEEDS=(42 137 2023 7 99)

# IMPORTANT: order must match -t 1-9
GROUP_LIST=(G1 G2 G3 G4 All Mechanical Electronic Thermal Other)

# Histogram output root
HIST_ROOT="results_hist_mt/${SMILES_SOURCE}"
BINS=100

# ===========================
#  SELECT GROUP FOR THIS TASK
# ===========================
IDX=$((SGE_TASK_ID-1))
GROUP="${GROUP_LIST[$IDX]}"

echo "[task ${SGE_TASK_ID}] group=${GROUP} source=${SMILES_SOURCE}"

[[ -f "${INPUT_CSV}" ]] || { echo "[ERROR] Missing ${INPUT_CSV}"; exit 1; }

BASE_DIR="RESULTS_${GROUP}/multitask_overall"
ENSEMBLE_CSV="${BASE_DIR}/ensemble_predictions_${SMILES_SOURCE}.csv"
OUT_DIR="${HIST_ROOT}/${GROUP}"

# ===========================
#  0) CLEAN old per-seed CSVs (optional safety)
# ===========================
for SEED in "${SEEDS[@]}"; do
  rm -f "${BASE_DIR}/seed_${SEED}/best_run/predictions_${SMILES_SOURCE}_seed_${SEED}.csv" || true
done

# ===========================
#  1) PREDICT (all seeds serial, skip missing)
# ===========================
for SEED in "${SEEDS[@]}"; do
  RUN_DIR="${BASE_DIR}/seed_${SEED}/best_run"
  OUT_CSV="${RUN_DIR}/predictions_${SMILES_SOURCE}_seed_${SEED}.csv"

  [[ -f "${RUN_DIR}/best.pt" ]] || continue
  [[ -f "${RUN_DIR}/target_scaler.pt" ]] || continue

  "$PYBIN" -u scripts/predict.py \
    --run_dir "${RUN_DIR}" \
    --input_csv "${INPUT_CSV}" \
    --fidelity "${FIDELITY}" \
    --batch_size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --out_csv "${OUT_CSV}"
done

# ===========================
#  2) AGGREGATE (streaming)
# ===========================
rm -f "${ENSEMBLE_CSV}"

if [[ -f "aggregate_seed_predictions.py" ]]; then
  "$PYBIN" -u aggregate_seed_predictions.py \
    --base_dir "${BASE_DIR}" \
    --seeds "42,137,2023,7,99" \
    --source "${SMILES_SOURCE}" \
    --pattern "predictions_${SMILES_SOURCE}_seed_{seed}.csv" \
    --chunksize "${CHUNKSIZE}" \
    --out_csv "${ENSEMBLE_CSV}"
else
  echo "[info] Missing aggregate_seed_predictions.py. Skipping aggregation and histogram generation."
  exit 0
fi

# ===========================
#  3) CLEANUP per-seed CSVs (keep only ensemble)
# ===========================
for SEED in "${SEEDS[@]}"; do
  rm -f "${BASE_DIR}/seed_${SEED}/best_run/predictions_${SMILES_SOURCE}_seed_${SEED}.csv" || true
done

# ===========================
#  4) HISTOGRAMS + STATS CSV (all mean_*)
# ===========================
[[ -f "${ENSEMBLE_CSV}" ]] || { echo "[ERROR] Missing ${ENSEMBLE_CSV}"; exit 1; }

mkdir -p "${OUT_DIR}"

"$PYBIN" -u scripts/plot_hist_mt.py \
  --csv "${ENSEMBLE_CSV}" \
  --out_dir "${OUT_DIR}" \
  --bins "${BINS}"

echo "[done] group=${GROUP} -> ${ENSEMBLE_CSV} and ${OUT_DIR}"
