#!/usr/bin/env bash
set -euo pipefail


SEEDS=(0 1 2)
DEVICE=0
MODEL="graph_transformer"

DATASETS=(
  "PROTEINS"
  "NCI1"
  "NCI109"
  "Mutagenicity"
  "PTC_FR"
  "PTC_FM"
  "PTC_MM"
  "PTC_MR"
)

#FREEZE_SPACES=("none", "A", "B", "C", "D", "all")
FREEZE_SPACES=("none")
MAX_DIM=2
MAX_RING_SIZE=6

LR=0.001
NUM_LAYERS=3
EMB_DIM=32
NUM_HEADS=8
BATCH_SIZE=128

USE_COBOUNDARIES=true
NONLINEARITY="relu"
JUMP_MODE="cat"
GRAPH_NORM="bn"
INIT_METHOD="mean"
ETA=0.1

MIX_DATASETS="PROTEINS,NCI1,NCI109,Mutagenicity,PTC_FM,PTC_FR,PTC_MM,PTC_MR"

PRETRAIN_RATIO=0.2
PROBE_RATIOS=(0.1)

PRETRAIN_EPOCHS=30
PROBE_BATCHES=20

# OVERLAP_PCTS=(10 20 30 40 50 60 70 80 90)

OVERLAP_PCTS=(20)

W_A=2
W_B=2
W_C=1
W_D=0.5

#EXP_TAG="second_order"
EXP_TAG="base"
RUNS_ROOT="runs"
STAMP="$(date +%Y%m%d_%H%M%S)"


MODE="pretrained"   # pretrain | pretrained
PRETRAINED_PT="runs/base__modepretrain__mgraph_transformer__eta0.1__20260203_034839/PROTEINS/seed0/checkpoints/pretrained.pt"
DO_TRAIN=true

PROBE_DATA_MODE="target"
HEATMAP_DATA_MODE="target"
TRAIN_DATA_MODE="target"


BASE_NAME="${EXP_TAG}__mode${MODE}__m${MODEL}__eta${ETA}__${STAMP}"
BATCH_DIR="${RUNS_ROOT}/${BASE_NAME}"
mkdir -p "${BATCH_DIR}"

echo "Batch dir:     ${BATCH_DIR}"
echo "Mode:          ${MODE}"
echo "Model:         ${MODEL}"
echo "Freeze:        ${FREEZE_SPACES[*]}"
echo "Seeds:         ${SEEDS[*]}"
echo "Datasets:      ${DATASETS[*]}"
echo "Probe ratios:  ${PROBE_RATIOS[*]}"
echo "Overlap pcts:  ${OVERLAP_PCTS[*]}  (CON==CE)"
echo

if [[ "${MODE}" == "pretrained" ]]; then
  if [[ ! -f "${PRETRAINED_PT}" ]]; then
    echo "[ERROR] PRETRAINED_PT not found: ${PRETRAINED_PT}" >&2
    exit 1
  fi
  echo "Using pretrained checkpoint: ${PRETRAINED_PT}"
  echo
fi

# flags
TRAIN_FLAGS=(--do_train)
if [[ "${DO_TRAIN}" != "true" ]]; then
  TRAIN_FLAGS=()
fi

PRETRAIN_FLAGS=(--do_pretrain)
if [[ "${MODE}" != "pretrain" ]]; then
  PRETRAIN_FLAGS=()
fi


for FREEZE in "${FREEZE_SPACES[@]}"; do
  echo "========================================"
  echo "==> ABLATION: freeze_space=${FREEZE}"
  echo "========================================"

  for DATASET in "${DATASETS[@]}"; do
    echo "=============================="
    echo "==> DATASET: ${DATASET}"
    echo "=============================="

    for PROBE_RATIO in "${PROBE_RATIOS[@]}"; do
      PR_TAG="${PROBE_RATIO/./p}"   # 0.05 -> 0p05
      echo "==> PROBE_RATIO: ${PROBE_RATIO} (tag=${PR_TAG})"

      for OVERLAP_PCT in "${OVERLAP_PCTS[@]}"; do
        OP_TAG="op${OVERLAP_PCT}"   # overlap pct tag
        echo "==> OVERLAP_PCT (CON==CE): ${OVERLAP_PCT} (tag=${OP_TAG})"

        for seed in "${SEEDS[@]}"; do
          EXP_NAME="${BASE_NAME}__freeze${FREEZE}__ds${DATASET}__pr${PR_TAG}__${OP_TAG}__seed${seed}"
          OUT_DIR="${BATCH_DIR}/${FREEZE}/${DATASET}/probe${PR_TAG}/${OP_TAG}/seed${seed}"
          LOG_FILE="${OUT_DIR}/train.log"
          mkdir -p "${OUT_DIR}/checkpoints"

          if [[ "${MODE}" == "pretrain" ]]; then
            CKPT_PATH="${OUT_DIR}/checkpoints/pretrained.pt"
          else
            CKPT_PATH="${PRETRAINED_PT}"
          fi

          echo "==> EXP_NAME: ${EXP_NAME}"
          echo "    OUT_DIR:  ${OUT_DIR}"
          echo "    LOG:      ${LOG_FILE}"
          echo "    CKPT:     ${CKPT_PATH}"
          echo

          python -m semiSupervised.simgrace_meta \
            --seed "${seed}" \
            --device "${DEVICE}" \
            --model "${MODEL}" \
            --use_coboundaries "${USE_COBOUNDARIES}" \
            --nonlinearity "${NONLINEARITY}" \
            --jump_mode "${JUMP_MODE}" \
            --graph_norm "${GRAPH_NORM}" \
            --lr "${LR}" \
            --num_layers "${NUM_LAYERS}" \
            --emb_dim "${EMB_DIM}" \
            --num_heads "${NUM_HEADS}" \
            --batch_size "${BATCH_SIZE}" \
            --dataset "${DATASET}" \
            --max_dim "${MAX_DIM}" \
            --max_ring_size "${MAX_RING_SIZE}" \
            --exp_name "${EXP_NAME}" \
            --init_method "${INIT_METHOD}" \
            --eta "${ETA}" \
            --ckpt_path "${CKPT_PATH}" \
            --mix_datasets "${MIX_DATASETS}" \
            --pretrain_ratio "${PRETRAIN_RATIO}" \
            --probe_ratio "${PROBE_RATIO}" \
            --pretrain_epochs "${PRETRAIN_EPOCHS}" \
            --probe_batches "${PROBE_BATCHES}" \
            --probe_data_mode "${PROBE_DATA_MODE}" \
            --heatmap_data_mode "${HEATMAP_DATA_MODE}" \
            --train_data_mode "${TRAIN_DATA_MODE}" \
            --overlap_pct_con "${OVERLAP_PCT}" \
            --overlap_pct_ce "${OVERLAP_PCT}" \
            --w_A "${W_A}" \
            --w_B "${W_B}" \
            --w_C "${W_C}" \
            --w_D "${W_D}" \
            --freeze_space "${FREEZE}" \
            "${PRETRAIN_FLAGS[@]}" \
            --do_probe \
            "${TRAIN_FLAGS[@]}" \
            --do_test \
            2>&1 | tee "${LOG_FILE}"
        done
      done
    done
  done
done

echo
echo "Done. Logs under: ${BATCH_DIR}"
