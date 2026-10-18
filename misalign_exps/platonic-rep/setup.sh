#!/bin/bash
# Fetch upstream platonic-rep's metrics.py and models.py into this directory.
# Usage: bash setup.sh
set -euo pipefail
COMMIT=dcd76ba3c950c1b197a2ae8b1c6713535c94ecf9
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git -C "$TMP" init -q
git -C "$TMP" remote add origin https://github.com/minyoungg/platonic-rep.git
git -C "$TMP" fetch -q --depth 1 origin "$COMMIT"
git -C "$TMP" checkout -q FETCH_HEAD
cp "$TMP/metrics.py" "$TMP/models.py" "$HERE/"
echo "Copied metrics.py and models.py (platonic-rep @ ${COMMIT:0:7}) into $HERE"
