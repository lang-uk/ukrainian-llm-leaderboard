---
title: Ukrainian LLM Leaderboard
emoji: 👁
colorFrom: red
colorTo: indigo
sdk: gradio
sdk_version: 6.2.0
app_file: leaderboard.py
pinned: false
license: mit
short_description: Measuring LLM capabilities to process Ukrainian texts
---

# 💪 Ukrainian LLM Leaderboard
        
This leaderboard displays performance metrics for language models on Ukrainian language benchmarks as used during development of [Lapa LLM](https://github.com/lapa-llm/lapa-llm).
The data comes from evaluation results [lang-uk/ukrainian-llm-leaderboard-results](https://huggingface.co/datasets/lang-uk/ukrainian-llm-leaderboard-results) . Locally it's stored in `eval-results/<model_name>/results*.json`.

## 📏 What does it measure?
The leaderboard evaluates models on a variety of benchmarks covering different NLP tasks in Ukrainian, including:
- 🌐 **Machine Translation**: FLORES-200 (en-uk, uk-en), LongFLORES (en-uk, uk-en), WMT-22 (en-uk, uk-en)
- 📌 **Summarization**: XLSUM (uk)
- 🔎 **In-Context Question Answering**: Belebele (uk), SQuAD (uk)
- 🤓 **Reasoning and Knowledge**: ZNO-Eval, Winogrande Challenge, Hellaswag, ARC Easy/Challenge, TriviaQA, MMLU
- 🔢 **Mathematical Problem Solving**: GSM-8K
- 👉 **Instruction Following**: IFEval

> **Note:** Summarization + Q&A is a proxy for RAG capabilities in Ukrainian.

## 🎯 Roadmap:

- [ ] Upload full benchmark traces
- [ ] Add [MMZNO](https://aclanthology.org/2025.unlp-1.2/) scoring for visual Q&A tasks
- [ ] Add [UAlign](https://aclanthology.org/2025.unlp-1.4/) for ethical alignment scoring
- [ ] Fix ZNO-Eval parsing issues
- [ ] Add tokenizer efficiency comparison
- [ ] Add parameter count
- [ ] Add API providers (OpenAI, Anthropic, Google)
- [ ] Add quantized model evaluations
- [ ] Add citation for each benchmark used

If you want to leave a feedback or suggest a new feature, please open an issue or a pull request here: https://github.com/lang-uk/ukrainian-llm-leaderboard

### Important Notes

- **FLORES benchmarks**: Only English↔Ukrainian (en-uk, uk-en) translation pairs are displayed
- **MMLU**: Only the aggregate score is shown (no subcategories)

### How to Use

- **Main Leaderboard**: View performance on core benchmarks
- **Detailed Benchmarks**: Explore performance on specific benchmark categories
- **Model Comparison**: Compare multiple models with radar charts
- **Visualizations**: Generate bar charts for specific metrics

Sort tables by any metric and adjust display options using the controls.

### ᯓ🏃🏻‍♀️‍➡️ How to Run Benchmarks

```bash
pip install -r requirements-evals.txt
```

Then run the evaluation script for each model checkpoint:

```bash
CHECKPOINT="le-llm/lapa-v0.1.2-instruct" 

VLLM_WORKER_MULTIPROC_METHOD=spawn  lm_eval --model vllm \
       --model_args pretrained=${CHECKPOINT},data_parallel_size=2,tensor_parallel_size=2,gpu_memory_utilization=0.6,add_bos_token=True,think_end_token='</think>',max_gen_toks=32000,max_length=65536 \
       --tasks ukrainian_bench \
        --gen_kwargs temperature=0.7,top_p=0.95,until='<asdasdgfggvvcccx>',max_gen_toks=32000 \
        --batch_size auto \
        --output_path ./eval-results \
        --log_samples \
        --include_path ./tasks \
        --apply_chat_template
```


## 📚 Citation

```
# main citation for the leaderboard 
# and benchmark suite: FLORES + LongFLORES + XLSUM + Belebele + SQuAD
@InProceedings{paniv:2025:RANLP,
  author    = {Paniv, Yurii},
  title     = {Isolating LLM Performance Gains in Pre-training versus Instruction-tuning for Mid-resource Languages: The Ukrainian Benchmark Study},
  booktitle      = {Proceedings of the 15th International Conference on Recent Advances in Natural Language Processing - Natural Language Processing in the Generative AI Era},
  month          = {September},
  year           = {2025},
  address        = {Varna, Bulgaria},
  publisher      = {INCOMA Ltd., Shoumen, Bulgaria},
  pages     = {876--883},
  abstract  = {This paper evaluates language model performance on Ukrainian language tasks across multiple downstream benchmarks, including summarization, closed and open question answering, and translation at both sentence and paragraph levels. We also introduce LongFlores, an extension of the FLORES benchmark designed specifically to assess paragraph-level translation capabilities. In our experiments, we compare the performance of base models against their instruction-tuned counterparts to isolate and quantify the source of performance improvements for Ukrainian language tasks. Our findings reveal that for popular open source models, base models are stronger in the few-shot setting for the task than their instruction-tuned counterparts in the zero-shot setting. This suggests lower attention paid to Ukrainian during the instruction-tuning phase, providing valuable insights for future model development and optimization for Ukrainian and potentially other lower-resourced languages.},
  url       = {https://aclanthology.org/2025.ranlp-1.100}
}

# MamayLM benchmarks suite: ZNO + Winogrande challenge + Hellaswag + ARC Easy/Challenge + TriviaQA +  GSM-8K + MMLU + IFEval

@misc{MamayLM,
  author = {Yukhymenko, Hanna and Alexandrov, Anton and Vechev, Martin},
  title = {MamayLM: An efficient state-of-the-art Ukrainian LLM},
  year = {2025},
  publisher = {INSAIT},
  howpublished = {https://huggingface.co/blog/INSAIT-Institute/mamaylm}
}

# if using ZNO-Eval benchmark results
@article{Syromiatnikov_2024,
   title={ZNO-Eval: Benchmarking reasoning capabilities of large  language models in Ukrainian},
   volume={1},
   ISSN={2522-1523},
   url={http://dx.doi.org/10.15276/ict.01.2024.27},
   DOI={10.15276/ict.01.2024.27},
   number={1},
   journal={Informatics. Culture. Technology},
   publisher={Odessa Polytechnic National University},
   author={Syromiatnikov, Mykyta V. and Ruvinskaya, Victoria M. and Troynina, Anastasiya S.},
   year={2024},
   month=sep, pages={186–191} }

```