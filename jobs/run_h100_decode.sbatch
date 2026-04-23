#!/bin/bash
#SBATCH -A cis260009p
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH -t 04:00:00
#SBATCH -J decode-h100
#SBATCH -o slurm-%x-%j.out

set -eo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT_DIR"

module load anaconda3
conda activate
conda activate "$ROOT_DIR/.conda_envs/attn_backend"
module load cuda

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RESULT_DIR="$ROOT_DIR/results/decode/$RUN_TAG"
mkdir -p "$RESULT_DIR"

echo "==== Job Metadata ====" | tee "$RESULT_DIR/job_info.txt"
echo "job_id=$SLURM_JOB_ID" | tee -a "$RESULT_DIR/job_info.txt"
echo "job_name=$SLURM_JOB_NAME" | tee -a "$RESULT_DIR/job_info.txt"
echo "node_list=$SLURM_JOB_NODELIST" | tee -a "$RESULT_DIR/job_info.txt"
echo "submit_dir=$ROOT_DIR" | tee -a "$RESULT_DIR/job_info.txt"
echo "result_dir=$RESULT_DIR" | tee -a "$RESULT_DIR/job_info.txt"
echo "start_time=$(date)" | tee -a "$RESULT_DIR/job_info.txt"

nvidia-smi -L | tee "$RESULT_DIR/nvidia_smi.txt"

echo "==== Environment Check ====" | tee "$RESULT_DIR/env_check.txt"
python - <<'PY' | tee -a "$RESULT_DIR/env_check.txt"
import torch
print("torch version:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

echo "==== Running Tests ====" | tee "$RESULT_DIR/tests.txt"
python -m pytest tests/test_attention_backends.py tests/test_minitorch_alignment.py tests/test_decode_kv.py -q | tee -a "$RESULT_DIR/tests.txt"

echo "==== Decode Benchmark ====" | tee "$RESULT_DIR/decode_benchmark.txt"
python benchmark_decode_kv.py \
  --device cuda \
  --initial-lens 512 1024 2048 4096 \
  --decode-steps 64 \
  --num-heads 8 \
  --head-dim 64 \
  --page-size 128 \
  --warmup-iters 5 \
  --measure-iters 20 \
  --output-json "$RESULT_DIR/decode_h100_bs4_p128.json" | tee -a "$RESULT_DIR/decode_benchmark.txt"

echo "==== Finished ====" | tee -a "$RESULT_DIR/job_info.txt"
echo "end_time=$(date)" | tee -a "$RESULT_DIR/job_info.txt"
