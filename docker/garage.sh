#!/usr/bin/env bash
set -euo pipefail

cmd="${1:-shell}"
if [[ $# -gt 0 ]]; then
  shift
fi

export PYTHONPATH="${PYTHONPATH:-/workspace}"
export LM_STUDIO_BASE_URL="${LM_STUDIO_BASE_URL:-http://lmstudio:1234/v1}"
export LM_STUDIO_MODEL="${LM_STUDIO_MODEL:-granite-4.0-h-micro-GGUF}"

EVAL_DATASET="${EVAL_DATASET:-llm/evaluation/shared_dataset/test.npz}"
TRAIN_DATASET_DIR="${TRAIN_DATASET_DIR:-llm/evaluation/shared_dataset}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-checkpoints/stage1_best_forecast_20260302_223838.pt}"
STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-checkpoints/stage2_clean_20260302_223838.pt}"

BASELINE_OUT="${BASELINE_OUT:-results-new/results_llm_baseline_docker.json}"
KAG_OUT="${KAG_OUT:-results-new/results_gdn_kg_llm_docker.json}"

case "$cmd" in
  dashboard)
    exec streamlit run demo/demo_app.py \
      --server.address 0.0.0.0 \
      --server.port 8501 \
      --server.headless true
    ;;

  build-cache)
    exec python demo/build_demo_cache.py \
      --checkpoint "$STAGE2_CHECKPOINT" \
      --data "$EVAL_DATASET" \
      --output demo/demo_cache.pkl \
      "$@"
    ;;

  eval-baseline)
    exec python llm/evaluation/evaluate_llm_baseline.py \
      --dataset "$EVAL_DATASET" \
      --model-path "$STAGE2_CHECKPOINT" \
      --model-repo "$LM_STUDIO_MODEL" \
      --base-url "$LM_STUDIO_BASE_URL" \
      --output "$BASELINE_OUT" \
      "$@"
    ;;

  eval-kag)
    exec python llm/evaluation/evaluate_gdn_kg_llm.py \
      --dataset "$EVAL_DATASET" \
      --model-path "$STAGE2_CHECKPOINT" \
      --model-repo "$LM_STUDIO_MODEL" \
      --base-url "$LM_STUDIO_BASE_URL" \
      --output "$KAG_OUT" \
      "$@"
    ;;

  eval-all)
    python llm/evaluation/evaluate_llm_baseline.py \
      --dataset "$EVAL_DATASET" \
      --model-path "$STAGE2_CHECKPOINT" \
      --model-repo "$LM_STUDIO_MODEL" \
      --base-url "$LM_STUDIO_BASE_URL" \
      --output "$BASELINE_OUT" \
      "$@"

    python llm/evaluation/evaluate_gdn_kg_llm.py \
      --dataset "$EVAL_DATASET" \
      --model-path "$STAGE2_CHECKPOINT" \
      --model-repo "$LM_STUDIO_MODEL" \
      --base-url "$LM_STUDIO_BASE_URL" \
      --output "$KAG_OUT" \
      "$@"
    ;;

  train-stage1)
    exec python training/train_stage1.py \
      --data_path "$TRAIN_DATASET_DIR" \
      "$@"
    ;;

  train-stage2)
    exec python training/train_stage2_clean.py \
      --data_path "$TRAIN_DATASET_DIR" \
      --stage1_checkpoint "$STAGE1_CHECKPOINT" \
      "$@"
    ;;

  shell)
    exec bash "$@"
    ;;

  *)
    cat <<USAGE
Unknown command: $cmd

Usage: garage.sh <command>
  dashboard      Run Streamlit dashboard on :8501
  build-cache    Rebuild demo/demo_cache.pkl from checkpoint
  eval-baseline  Run LLM baseline eval on test split
  eval-kag       Run GDN+KG+LLM eval on test split
  eval-all       Run both eval pipelines
  train-stage1   Retrain stage 1 model
  train-stage2   Retrain stage 2 model
  shell          Open a bash shell
USAGE
    exit 1
    ;;
esac
