#!/usr/bin/env bash
set -euo pipefail

export PATH="/usr/local/bin:${PATH}"

MODEL_REPO="ibm-granite/granite-4.0-h-micro-GGUF"
MODEL_HF_URL="https://huggingface.co/${MODEL_REPO}"
MODEL_QUANT_RAW="${LM_STUDIO_QUANTIZATION:-Q4_K_M}"
MODEL_QUANT="$(printf "%s" "$MODEL_QUANT_RAW" | tr '[:lower:]' '[:upper:]')"
MODEL_DOWNLOAD_KEY="${MODEL_HF_URL}@${MODEL_QUANT}"
MODEL_IDENTIFIER="${LM_STUDIO_MODEL_IDENTIFIER:-granite-4.0-h-micro-GGUF}"
MODEL_KEY_FALLBACK="${LM_STUDIO_MODEL_KEY:-granite-4.0-h-micro}"
PORT="${LM_STUDIO_PORT:-1234}"
CONTEXT_LENGTH="${LM_STUDIO_CONTEXT_LENGTH:-32768}"
GPU_OFFLOAD="${LM_STUDIO_GPU_OFFLOAD:-off}"

if [[ "${LM_STUDIO_MODEL_REPO:-$MODEL_REPO}" != "$MODEL_REPO" ]]; then
  echo "Error: only $MODEL_REPO is supported by this stack."
  exit 1
fi

echo "[lmstudio] Starting daemon..."
lms daemon up

echo "[lmstudio] Downloading model ${MODEL_DOWNLOAD_KEY} (if needed)..."
lms get "${MODEL_DOWNLOAD_KEY}" --gguf

echo "[lmstudio] Removing non-Granite bundled/local models..."
rm -rf /root/.lmstudio/.internal/bundled-models || true
if [[ -d /root/.lmstudio/models ]]; then
  find /root/.lmstudio/models -mindepth 1 -maxdepth 1 ! -name "ibm-granite" -exec rm -rf {} +
fi

MODEL_KEY="$(
  lms ls --json | jq -r --arg repo "$MODEL_REPO" --arg quant "$MODEL_QUANT" '
    map(select(.type == "llm"))
    | map(select((.path // "") | contains($repo)))
    | map(select((((.quantization.name // "") | ascii_upcase) == $quant)))
    | .[0].modelKey // empty
  '
)"

if [[ -z "${MODEL_KEY}" ]]; then
  MODEL_KEY="$(
    lms ls --json | jq -r --arg fallback "$MODEL_KEY_FALLBACK" '
      map(select(.type == "llm"))
      | map(select((.modelKey // "") == $fallback))
      | .[0].modelKey // empty
    '
  )"
fi

if [[ -z "${MODEL_KEY}" ]]; then
  echo "Error: unable to resolve downloaded Granite model key from 'lms ls --json'."
  lms ls --json || true
  exit 1
fi

echo "[lmstudio] Loading model key '${MODEL_KEY}' with identifier '${MODEL_IDENTIFIER}'..."
lms load "${MODEL_KEY}" \
  --identifier "${MODEL_IDENTIFIER}" \
  --context-length "${CONTEXT_LENGTH}" \
  --gpu "${GPU_OFFLOAD}"

echo "[lmstudio] Starting HTTP server on port ${PORT}..."
lms server start --port "${PORT}" --bind "0.0.0.0" --cors

echo "[lmstudio] Server started; entering supervision loop."
while true; do
  if ! curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    echo "[lmstudio] HTTP server became unavailable; exiting."
    exit 1
  fi
  sleep 10
done
