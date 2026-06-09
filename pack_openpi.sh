#!/bin/bash
set -euo pipefail

tar -czf openpi.tar.gz \
    --exclude='openpi/.venv' \
    --exclude='openpi/.uv_cache' \
    --exclude='openpi/.hf_cache' \
    --exclude='openpi/.openpi_cache' \
    --exclude='openpi/checkpoints' \
    openpi/

echo "Done: openpi.tar.gz ($(du -sh openpi.tar.gz | cut -f1))"
