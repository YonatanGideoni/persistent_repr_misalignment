#!/bin/bash
# Submit all 26 fine-tuned-backbone jobs
set -euo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$JOB_DIR/../.." && pwd)"
cd "$REPO_ROOT"

for job_file in slurm_scripts/vit_finetunes/*.job; do
    echo "submitting $job_file"
    sbatch "$job_file"
done
