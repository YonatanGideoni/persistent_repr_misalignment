#!/bin/bash
# Creates the ktt_llava conda env (LLaVA training + evaluation) and sets up the patched lmms-eval and
# Bongard-in-Wonderland clones in this directory.
# Usage: bash llava/setup_env.sh   [ENV_NAME=... to change the env name]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ENV_NAME=${ENV_NAME:-ktt_llava}

eval "$(conda shell.bash hook)"
conda create -n "$ENV_NAME" python=3.10 -y
conda activate "$ENV_NAME"
pip install -e .
pip install ninja  # must exist before flash-attn compiles, else the build is serial (hours)
pip install -e ".[train]" --no-build-isolation  # flash-attn builds against the torch installed above

# lmms-eval (MMVP, VLMBias, OmniSpatial), pinned + patched to accept model_base for LoRA checkpoints
if [ ! -d lmms-eval ]; then
    git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
    (cd lmms-eval && git checkout d6cc2b56 && git apply ../patches/lmms-eval.patch)
fi
pip install -e lmms-eval
pip install -e .  # re-pin transformers==4.37.2 (lmms-eval declares >=4.39.2; pip will warn)

# Bongard-in-Wonderland, pinned + patched; must be cloned as bongard/ in this directory
if [ ! -d bongard ]; then
    git clone https://github.com/ml-research/bongard-in-wonderland.git bongard
    (cd bongard && git checkout 48f042b && git apply ../patches/bongard-in-wonderland.patch)
fi
pip install anthropic openai google-generativeai pyyaml rtpt==0.0.4  # imported at module level by bongard, never called

echo "Done. Download the Bongard problems into bongard/data/bpimgs (see bongard/README.md, 'Download the data')."
