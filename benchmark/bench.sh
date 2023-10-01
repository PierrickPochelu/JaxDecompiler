#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:${PWD}"
benchpath="./benchmark/"
cd ${benchpath}

python3 MLP.py 1024 1024 2
python3 MLP.py 16 1024 128
python3 MLP.py 16 8192 2

N=32000000
python3 MapReduce.py 1 $N
python3 MapReduce.py 2 $N
python3 MapReduce.py 128 $N

