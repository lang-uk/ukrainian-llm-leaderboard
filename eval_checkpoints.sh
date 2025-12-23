# make for loop to run eval_checkpoints.sh for each checkpoint in the checkpoints directory
# array with all checkpoints in the checkpoints directory: dir1, google/gemma-3-12b-it, dir2
#!/bin/bash

CHECKPOINTS=(
"le-llm/lapa-v0.1.2-instruct"
"INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v1.0"
"google/gemma-3-12b-it"
"google/gemma-3-4b-it"
"Qwen/Qwen3-8B"
"meta-llama/Llama-3.1-8B-Instruct"
#"google/gemma-3-27b-it"
)
for CHECKPOINT in "${CHECKPOINTS[@]}"
do
   echo "Evaluating checkpoint: ${CHECKPOINT}"
   VLLM_WORKER_MULTIPROC_METHOD=spawn  lm_eval --model vllm \
       --model_args pretrained=${CHECKPOINT},data_parallel_size=2,tensor_parallel_size=2,gpu_memory_utilization=0.6,add_bos_token=True,think_end_token='</think>',max_gen_toks=32000,max_length=65536 \
       --tasks ukrainian_bench \
        --gen_kwargs temperature=0.7,top_p=0.95,until='<asdasdgfggvvcccx>',max_gen_toks=32000 \
        --batch_size auto \
        --output_path ./eval-results \
        --log_samples \
        --include_path ./tasks \
        --apply_chat_template # \
        #--num_fewshot 3
done 

# flores_en-uk,long_flores_en-uk,ifeval_uk,wmt_en_uk
# flores_en-uk,long_flores_en-uk,ifeval_uk 
# wmt_en_uk,flores_en-uk,long_flores_en-uk,ifeval_uk
# belebele_ukr_Cyrl, xlsum_uk, flores_en-uk,  squad_uk,

# disabled until fixed
# zno_uk_geography,zno_uk_history,zno_uk_language_and_literature,zno_uk_math