#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <model_id> [model_id ...]" >&2
    exit 1
fi

TASKS="ukrainian_bench_reasoning_api"
INCLUDE_PATH="./tasks/ukrainian_bench_reasoning"
OUTPUT_PATH="./eval-results-reasoning"
GEN_KWARGS="temperature=0.7,top_p=0.95,until=<asdasdgfggvvcccx>,skip_special_tokens=False"

export HF_DATASETS_TRUST_REMOTE_CODE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

for CHECKPOINT in "$@"
do
   MODEL_ARGS="pretrained=${CHECKPOINT},data_parallel_size=1,tensor_parallel_size=1,gpu_memory_utilization=0.90,max_model_len=131072,max_num_seqs=128,max_num_batched_tokens=131072,add_bos_token=True,enable_thinking=True,trust_remote_code=True"
   if [[ -n "${REASONING_PARSER:-}" ]]; then
       MODEL_ARGS="${MODEL_ARGS},reasoning_parser=${REASONING_PARSER}"
   fi

   echo "Evaluating checkpoint: ${CHECKPOINT}"
   echo "Tasks: ${TASKS}"
   echo "GEN_KWARGS=${GEN_KWARGS}"
   lm_eval run \
       --model vllm \
       --model_args "${MODEL_ARGS}" \
       --tasks ${TASKS} \
       --batch_size auto \
       --gen_kwargs "${GEN_KWARGS}" \
       --output_path "${OUTPUT_PATH}" \
       --log_samples \
       --include_path "${INCLUDE_PATH}" \
       --apply_chat_template
done
