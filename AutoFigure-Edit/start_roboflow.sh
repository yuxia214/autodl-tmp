#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_ACTIVATE="$ROOT_DIR/figure/bin/activate"

load_export_from_bashrc() {
  local var_name="$1"
  local bashrc="${HOME}/.bashrc"
  local line value

  if [ -n "${!var_name:-}" ]; then
    return
  fi
  if [ ! -f "$bashrc" ]; then
    return
  fi

  line="$(grep -E "^export ${var_name}=" "$bashrc" | tail -n 1 || true)"
  if [ -z "$line" ]; then
    return
  fi

  value="${line#*=}"
  value="${value%\"}"
  value="${value#\"}"
  value="${value%\'}"
  value="${value#\'}"
  if [ -n "$value" ]; then
    export "${var_name}=${value}"
  fi
}

if [ -f /etc/network_turbo ]; then
  # Optional network acceleration for GitHub/Hugging Face/API endpoints.
  # shellcheck disable=SC1091
  source /etc/network_turbo
fi

if [ ! -f "$ENV_ACTIVATE" ]; then
  echo "Virtualenv not found: $ENV_ACTIVATE"
  echo "Expected environment path: $ROOT_DIR/figure"
  exit 1
fi

# shellcheck disable=SC1090
source "$ENV_ACTIVATE"

load_export_from_bashrc "ROBOFLOW_API_KEY"
load_export_from_bashrc "BIANXIE_API_KEY"
load_export_from_bashrc "OPENROUTER_API_KEY"
load_export_from_bashrc "CRS_API_KEY"
: "${ROBOFLOW_API_KEY:=Vxe4NqybbwczubYJyMP4}"
export ROBOFLOW_API_KEY
: "${CRS_API_KEY:=sk-HMR2NAznJcxT122qSoDie0uNgbmb6OeDJeKEkj08HtWo5h2R}"
export CRS_API_KEY

if [ -z "${ROBOFLOW_API_KEY:-}" ]; then
  echo "ROBOFLOW_API_KEY is not set."
  echo "Set it first, e.g.:"
  echo "  export ROBOFLOW_API_KEY='your_key_here'"
  exit 1
fi

cd "$SCRIPT_DIR"
exec python server.py
