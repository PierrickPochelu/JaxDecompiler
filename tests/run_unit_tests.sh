#!/bin/sh
JAX_DECOMPILER_PATH="."  # <--- WARNING: you must be in the src/ for running it
PYTHONPATH="${PYTHONPATH}:${JAX_DECOMPILER_PATH}/src/"
python3 ${JAX_DECOMPILER_PATH}/tests/unit_test.py
