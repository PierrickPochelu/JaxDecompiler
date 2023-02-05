#!/bin/sh
# requirement: package coverage
JAX_DECOMPILER_PATH="." # <--- WARNING: you must be in the folder of the Python project (containing src/, tests/, ...)
export PYTHONPATH="${PYTHONPATH}:${JAX_DECOMPILER_PATH}/src/"
coverage run  ${JAX_DECOMPILER_PATH}/tests/unit_test.py
coverage html --omit=${JAX_DECOMPILER_PATH}/tests/test*

