#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PARENT="$(dirname "$SCRIPT_DIR")"
OUTPUT="$REPO_PARENT/openpi.tar.gz"

tar -czf "$OUTPUT" \
    -C "$REPO_PARENT" \
    --exclude='openpi/.venv' \
    --exclude='openpi/.uv_cache' \
    --exclude='openpi/.hf_cache' \
    --exclude='openpi/.openpi_cache' \
    --exclude='openpi/checkpoints' \
    openpi/

echo "Done: $OUTPUT ($(du -sh "$OUTPUT" | cut -f1))"
