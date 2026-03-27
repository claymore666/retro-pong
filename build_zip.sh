#!/usr/bin/env bash
# Build a portable zipapp (requires numpy installed on target)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$SCRIPT_DIR/retro-pong.pyz"

mkdir -p /tmp/retropong-build
cp "$SCRIPT_DIR/pong.py" /tmp/retropong-build/__main__.py
python3 -m zipapp /tmp/retropong-build \
    --output "$OUT" \
    --python "/usr/bin/env python3"
rm -rf /tmp/retropong-build

chmod +x "$OUT"
echo "Built: $OUT"
echo "Run with: python3 $OUT"
