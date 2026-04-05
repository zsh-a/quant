#!/usr/bin/env bash
# Neural formula search (AlphaGPT-style Transformer + REINFORCE)
#
# Usage:
#   ./scripts/run_neural_search.sh                         # defaults
#   ./scripts/run_neural_search.sh --generations 200       # override any arg
#   NEURAL_BATCH=8192 ./scripts/run_neural_search.sh       # env override
#
# Training output:
#   data/alpha_lab/neural/training_history.json
#   data/alpha_lab/neural/training_curves.png
#
# All extra arguments are forwarded to `python -m src.alpha.cli search`.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# ── Defaults (override via env) ──────────────────────────────────────────
SYMBOLS="${SYMBOLS:-BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT}"
INTERVAL="${INTERVAL:-5m}"
START="${START:-2022-01-01T00:00:00}"
END="${END:-2025-07-01T00:00:00}"
GENERATIONS="${GENERATIONS:-100}"
POP_SIZE="${POP_SIZE:-8}"
OFFSPRING="${OFFSPRING:-4}"
TOP_K="${TOP_K:-10}"
N_SPLITS="${N_SPLITS:-5}"
NEURAL_BATCH="${NEURAL_BATCH:-4096}"
LLM_BACKEND="${LLM_BACKEND:-heuristic}"

exec python -m src.alpha.cli search \
  --symbols "${SYMBOLS}" \
  --interval "${INTERVAL}" \
  --start "${START}" \
  --end "${END}" \
  --generations "${GENERATIONS}" \
  --population-size "${POP_SIZE}" \
  --offspring-count "${OFFSPRING}" \
  --top-k "${TOP_K}" \
  --n-splits "${N_SPLITS}" \
  --llm-backend "${LLM_BACKEND}" \
  --strategy neural \
  --neural-batch "${NEURAL_BATCH}" \
  "$@"
