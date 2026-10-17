#!/usr/bin/env bash
set -euo pipefail

# Persistent production root on AutoDL. Keep the served code under /opt, not /root.
ROOT="${ROOT:-/opt/molink/hanging}"
CONDA_ENV="${CONDA_ENV:-metric3d}"
HANGING_MVP_PORT="${HANGING_MVP_PORT:-6010}"

source /opt/molink/app/.venv/bin/activate

cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

# Optional BFL/VLM env. The hanging service does not call BFL, but the exit gate
# may reuse VLM env vars and downstream scripts share the same env file.
if [ -f /etc/molink/gpu.env ]; then
  source /etc/molink/gpu.env
  export VLM_PROVIDER VLM_API_URL VLM_API_KEY VLM_MODEL
  export SPATIAL_VLM_API_URL SPATIAL_VLM_API_KEY SPATIAL_VLM_MODEL SPATIAL_VLM_TIMEOUT_S
  export API_KEY ALLOW_UNAUTH_LOCAL
  export R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY R2_PUBLIC_BASE_URL
  export APIYI_API_KEY APIYI_API_BASE APIYI_GPT_IMAGE_MODEL APIYI_GPT_IMAGE_TIMEOUT_S
  export BFL_API_KEY HANGING_RENDER_PROVIDER
  export NO_PROXY HTTP_PROXY HTTPS_PROXY
  export OMP_NUM_THREADS
fi
[ -f .env.local ] && source .env.local
if [ -f "$ROOT/.env.bfl" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/.env.bfl"
fi

# Local dev: allow unauthenticated requests and set API_KEY from VLM_API_KEY
if [ -f "$ROOT/.env" ]; then
  source "$ROOT/.env"
fi
export ALLOW_UNAUTH_LOCAL="${ALLOW_UNAUTH_LOCAL:-1}"
export MOLINK_SKIP_ARTWORK_PROTECTION="${MOLINK_SKIP_ARTWORK_PROTECTION:-1}"
if [ -z "${API_KEY:-}" ] && [ -n "${VLM_API_KEY:-}" ]; then
  export API_KEY="$VLM_API_KEY"
fi

exec /opt/molink/app/.venv/bin/python -m uvicorn hanging_mvp.service:app --host 0.0.0.0 --port "$HANGING_MVP_PORT"