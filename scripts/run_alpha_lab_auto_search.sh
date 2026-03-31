#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-${ROOT_DIR}/config/alpha_lab/auto_search.bulk.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${ROOT_DIR}"
exec "${PYTHON_BIN}" -m src.alpha.cli auto-search-db --config "${CONFIG_PATH}"
