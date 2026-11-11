#!/usr/bin/env bash
cd "$(dirname "$0")"; mkdir -p input output
[ -x .venv/bin/python ] && .venv/bin/python -c "import numpy,PIL" 2>/dev/null || { rm -rf .venv; python3 -m venv .venv && .venv/bin/pip install -q -r scripts/requirements.txt; }
.venv/bin/pip install -q -r scripts/requirements-optional.txt 2>/dev/null
.venv/bin/python scripts/batch.py
