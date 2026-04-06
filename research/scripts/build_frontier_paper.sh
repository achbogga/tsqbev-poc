#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_REL="${1:-docs/paper/tsqbev_frontier_proposal.tex}"
SOURCE="$ROOT/$SOURCE_REL"
OUTDIR="$ROOT/docs/paper/build"

mkdir -p "$OUTDIR"

if command -v tectonic >/dev/null 2>&1; then
  tectonic --outdir "$OUTDIR" "$SOURCE"
elif [ -x "$HOME/.local/bin/tectonic" ]; then
  "$HOME/.local/bin/tectonic" --outdir "$OUTDIR" "$SOURCE"
else
  echo "tectonic is not installed or not on PATH" >&2
  exit 1
fi

PDF_NAME="$(basename "${SOURCE_REL%.tex}.pdf")"
cp "$OUTDIR/$PDF_NAME" "$ROOT/docs/paper/$PDF_NAME"
echo "built $ROOT/docs/paper/$PDF_NAME"
