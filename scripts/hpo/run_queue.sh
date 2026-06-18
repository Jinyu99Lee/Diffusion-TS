#!/usr/bin/env bash
# Dispatch the generated ILI HPO scripts across GPUs, keeping each card busy with
# exactly one sub-experiment at a time. Polls GPU memory; when a card is free it
# launches the next pending script on it. Each launched script self-detaches
# (nohup &), so "free" is detected via memory use, not process tracking.
#
# Usage:
#   ./scripts/hpo/run_queue.sh ["GPU_LIST"] [MEM_THRESH_MIB] [MANIFEST]
# Examples:
#   ./scripts/hpo/run_queue.sh                 # GPUs "0 1 2", thresh 2000
#   ./scripts/hpo/run_queue.sh "0 1"           # only cards 0 and 1
#   ./scripts/hpo/run_queue.sh "0 1 2" 4000    # treat <4000MiB used as free
#
# Recommended: run detached so a dropped connection doesn't kill it, e.g.
#   nohup ./scripts/hpo/run_queue.sh "0 1 2" > Logs/hpo/queue_$(date +%F_%H-%M-%S).log 2>&1 &
set -euo pipefail

REPO=/data/jinyuli/Projects/Diffusion-TS
cd "$REPO"

GPUS_STR="${1:-0 1 2}"
THRESH="${2:-2000}"                       # MiB; a card using less than this is "free"
MANIFEST="${3:-scripts/hpo/ili/manifest.txt}"
POLL="${POLL:-30}"                        # seconds between polls / after a launch

read -ra GPUS <<< "$GPUS_STR"
[ -f "$MANIFEST" ] || { echo "manifest not found: $MANIFEST" >&2; exit 1; }
mapfile -t S < <(grep -v '^[[:space:]]*$' "$MANIFEST")

echo "[queue] ${#S[@]} script(s) | GPUs: ${GPUS[*]} | free-threshold: ${THRESH}MiB | poll: ${POLL}s"

i=0
while (( i < ${#S[@]} )); do
  for g in "${GPUS[@]}"; do
    (( i >= ${#S[@]} )) && break
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" | tr -d ' ')
    if (( used < THRESH )); then
      script="${S[$i]}"
      echo "[dispatch $((i+1))/${#S[@]}] $script -> GPU $g (used=${used}MiB)"
      if [ -x "$script" ]; then "./$script" "$g"; else bash "$script" "$g"; fi
      i=$((i + 1))                        # NOT ((i++)): that returns 1 when i=0 and set -e would exit
      sleep "$POLL"                       # let it grab the card before re-polling
    fi
  done
  (( i < ${#S[@]} )) && sleep "$POLL"
done

echo "[queue] all ${#S[@]} script(s) dispatched. (training continues in background;"
echo "        watch with: nvidia-smi  and  tail -f Logs/hpo/ili/**/*.log)"
