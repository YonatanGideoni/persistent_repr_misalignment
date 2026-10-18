#!/bin/bash
# Creates the misalign_exps conda env, used for everything outside llava/
# Usage: bash misalign_exps/setup_env.sh   [ENV_NAME=... to change the env name]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ENV_NAME=${ENV_NAME:-misalign_exps}

eval "$(conda shell.bash hook)"
conda create -n "$ENV_NAME" python=3.10 -y
conda activate "$ENV_NAME"
pip install -r requirements.txt

# upstream platonic-rep metrics.py/models.py for the representational-alignment scores
bash platonic-rep/setup.sh
