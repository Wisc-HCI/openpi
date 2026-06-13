#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PARENT="$(dirname "$SCRIPT_DIR")"
OUTPUT="$REPO_PARENT/openpi.tar.gz"

# Download uv binary if not present (needed on CHTC nodes which don't have uv)
UV_BIN="$SCRIPT_DIR/bin/uv"
UV_VERSION="0.11.18"
if [ ! -f "$UV_BIN" ]; then
    echo "Downloading uv $UV_VERSION..."
    mkdir -p "$SCRIPT_DIR/bin"
    curl -fsSL "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz" \
        | tar -xz -C "$SCRIPT_DIR/bin" --strip-components=1 "uv-x86_64-unknown-linux-gnu/uv"
    chmod +x "$UV_BIN"
    echo "uv downloaded: $("$UV_BIN" --version)"
fi

tar -czf "$OUTPUT" \
    -C "$REPO_PARENT" \
    --exclude='openpi/.venv' \
    --exclude='openpi/.uv_cache' \
    --exclude='openpi/.hf_cache' \
    --exclude='openpi/.openpi_cache' \
    --exclude='openpi/checkpoints' \
    openpi/

echo "Done: $OUTPUT ($(du -sh "$OUTPUT" | cut -f1))"
