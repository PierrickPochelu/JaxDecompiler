#!/bin/sh
# requirement: package coverage
JAX_DECOMPILER_PATH="."
coverage run  ${JAX_DECOMPILER_PATH}/tests/unit_test.py
coverage html --omit=${JAX_DECOMPILER_PATH}/tests/test*

