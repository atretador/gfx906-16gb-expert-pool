#!/usr/bin/env bash
#
# Example launcher for the gfx906-16gb-expert-pool fork.
#
# Validated on an MI50 (gfx906, 16 GB) with a 128K context. Every value can be
# overridden from the environment. Read the README before changing the pool
# knobs: the slot count and the rail are what decide whether the pool fits.
#
# Usage:
#   MODEL_DIR=/path/to/models ./scripts/run-example.sh
#
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-./models}"
MODEL="${MODEL:-Qwen3.8-Flash-Next-AD-3.84bpw-IQ4_XS-M64-00001-of-00028.gguf}"
MMPROJ="${MMPROJ:-mmproj-Qwen3.8-Flash-Next-F16.gguf}"
IMAGE="${IMAGE:-local/llama.cpp-gfx906:expert-pool}"
CONTAINER_NAME="${CONTAINER_NAME:-llamacpp-expert-pool}"
PORT="${PORT:-8886}"

CONTEXT="${CONTEXT:-131072}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q8_0}"
BATCH="${BATCH:-2048}"
UBATCH="${UBATCH:-512}"
THREADS="${THREADS:-18}"
CACHE_RAM="${CACHE_RAM:-4096}"
CTX_CHECKPOINTS="${CTX_CHECKPOINTS:-2}"

# Offload all 48 MoE layers and keep a 66-slot pool per offloaded expert weight
# tensor. This is the validated 128K profile: about 5.3 GiB of pool inside a
# 16 GB card, roughly 15.0 GiB total VRAM in use.
N_CPU_MOE="${N_CPU_MOE:-48}"
EXPERT_CACHE="${EXPERT_CACHE:-66}"
POOL_RAIL_MIB="${POOL_RAIL_MIB:-2048}"
POOL_CAP_MIB="${POOL_CAP_MIB:-}"
POOL_PROFILE="${POOL_PROFILE:-}"

MODEL_PATH="$MODEL_DIR/$MODEL"
MMPROJ_PATH="$MODEL_DIR/$MMPROJ"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "[ERROR] Model not found: $MODEL_PATH"
  exit 1
fi
if [[ ! -f "$MMPROJ_PATH" ]]; then
  echo "[ERROR] Multimodal projector not found: $MMPROJ_PATH"
  exit 1
fi

POOL_ENV=(-e "LLAMA_MOE_POOL_RAIL_MIB=$POOL_RAIL_MIB")
if [[ -n "$POOL_CAP_MIB" ]]; then
  POOL_ENV+=(-e "LLAMA_MOE_POOL_CAP_MIB=$POOL_CAP_MIB")
fi
if [[ -n "$POOL_PROFILE" ]]; then
  POOL_ENV+=(-e "LLAMA_MOE_POOL_PROFILE=$POOL_PROFILE")
fi

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run -d \
  --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  -e HSA_OVERRIDE_GFX_VERSION=9.0.6 \
  -e GGML_MOE_POOL_STATS=1 \
  "${POOL_ENV[@]}" \
  -p "$PORT:8080" \
  -v "$MODEL_DIR:/models:ro" \
  "$IMAGE" \
  --host 0.0.0.0 \
  --port 8080 \
  -m "/models/$MODEL" \
  --mmproj "/models/$MMPROJ" \
  -ngl 99 \
  --n-cpu-moe "$N_CPU_MOE" \
  -c "$CONTEXT" \
  -b "$BATCH" \
  -ub "$UBATCH" \
  --threads "$THREADS" \
  --threads-batch "$THREADS" \
  --cache-ram "$CACHE_RAM" \
  --ctx-checkpoints "$CTX_CHECKPOINTS" \
  --cache-type-k "$CACHE_TYPE_K" \
  --cache-type-v "$CACHE_TYPE_V" \
  --flash-attn on \
  --load-mode none \
  --metrics \
  --jinja \
  --moe-expert-cache "$EXPERT_CACHE"

echo "[+] Container:  $CONTAINER_NAME (port $PORT)"
echo "[+] Pool:       $EXPERT_CACHE slots, rail ${POOL_RAIL_MIB} MiB"
echo "[i] Wait for health, then check the pool markers:"
echo "    docker logs $CONTAINER_NAME 2>&1 | grep 'expert pool'"
echo "[i] Watch copy_bytes in the shutdown/interval line, not only the hit rate."
