#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

from datasets import Dataset
from datasets import load_dataset
from lm_eval import utils as lm_eval_utils
from lm_eval.api.instance import Instance
from lm_eval.api.registry import get_model


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


TASK_SPECS = {
    "global_mmlu_full_uk": {
        "dataset_path": "CohereForAI/Global-MMLU",
        "dataset_name": "uk",
        "split": "test",
        "output": "cot_traces/global_mmlu_full_uk/global_mmlu_full_uk_cot.parquet",
    },
    "belebele_ukr_Cyrl": {
        "dataset_path": "facebook/belebele",
        "dataset_name": "ukr_Cyrl",
        "split": "test",
        "output": "cot_traces/belebele_ukr_Cyrl/belebele_ukr_Cyrl_cot.parquet",
    },
    "arc_easy_uk": {
        "dataset_path": "INSAIT-Institute/arc-easy_ukr",
        "dataset_name": None,
        "split": "test",
        "output": "cot_traces/arc_easy_uk/arc_easy_uk_cot.parquet",
    },
    "winogrande_uk": {
        "dataset_path": "INSAIT-Institute/winogrande_ukr",
        "dataset_name": None,
        "split": "validation",
        "output": "cot_traces/winogrande_uk/winogrande_uk_cot.parquet",
    },
    "squad_uk": {
        "dataset_path": "FIdo-AI/ua-squad",
        "dataset_name": None,
        "split": "validation",
        "output": "cot_traces/squad_uk/squad_uk_cot.parquet",
    },
}


def labeled_options(labels: list[str], choices: list[str]) -> str:
    return "\n".join(f"{label}. {choice}" for label, choice in zip(labels, choices))


def winogrande_choices(doc: dict) -> list[str]:
    idx = doc["sentence"].index("_")
    return [
        doc["sentence"][:idx] + doc["option1"],
        doc["sentence"][:idx] + doc["option2"],
    ]


def build_user_prompt(task: str, doc: dict) -> str:
    if task == "global_mmlu_full_uk":
        choices = labeled_options(
            ["A", "B", "C", "D"],
            [doc["option_a"], doc["option_b"], doc["option_c"], doc["option_d"]],
        )
        answer_format = "<answer>LETTER</answer>, де LETTER - A, B, C або D"
        return (
            f"{doc['question'].strip()}\n"
            f"{choices}\n\n"
            "Поміркуй українською, який варіант є правильною відповіддю. "
            f"Після міркування дай відповідь у форматі {answer_format}."
        )

    if task == "belebele_ukr_Cyrl":
        choices = labeled_options(
            ["1", "2", "3", "4"],
            [doc["mc_answer1"], doc["mc_answer2"], doc["mc_answer3"], doc["mc_answer4"]],
        )
        return (
            f"P: {doc['flores_passage']}\n"
            f"Q: {doc['question'].strip()}\n"
            f"{choices}\n\n"
            "Поміркуй українською, який варіант є правильною відповіддю. "
            "Після міркування дай відповідь у форматі <answer>N</answer>, "
            "де N - номер правильного варіанта."
        )

    if task == "arc_easy_uk":
        labels = doc["choices"]["label"]
        choices = doc["choices"]["text"]
        return (
            f"Питання: {doc['question']}\n"
            f"{labeled_options(labels, choices)}\n\n"
            "Поміркуй українською, який варіант є правильною відповіддю. "
            "Після міркування скопіюй текст правильної відповіді у форматі "
            "<answer>текст відповіді</answer>."
        )

    if task == "winogrande_uk":
        choices = [doc["option1"], doc["option2"]]
        return (
            f"Речення з пропуском: {doc['sentence']}\n"
            f"{labeled_options(['1', '2'], choices)}\n\n"
            "Поміркуй українською, який варіант правильно заповнює пропуск. "
            "Після міркування скопіюй правильний варіант у форматі "
            "<answer>варіант</answer>."
        )

    if task == "squad_uk":
        return (
            f"Контекст: {doc['Context']}\n\n"
            f"Питання: {doc['Question']}\n\n"
            "Знайди найкоротший фрагмент з контексту, який відповідає на питання.\n"
            "Якщо відповідь не підтримується контекстом, виведи "
            "<answer>немає відповіді</answer>.\n"
            "Інакше виведи <answer>фрагмент з контексту</answer>."
        )

    raise ValueError(f"Unsupported task: {task}")


def build_model_args(args) -> dict:
    model_args = lm_eval_utils.simple_parse_args_string(args.model_args)
    # Trace generation needs raw reasoning tokens, not parsed final answers.
    model_args.pop("reasoning_parser", None)
    model_args.pop("think_end_token", None)
    return model_args


def load_task_dataset(task: str):
    if task == "squad_uk":
        from tasks.ukrainian_bench.squad_uk.task import SQuAD2

        squad_task = SQuAD2()
        squad_task.download()
        return Dataset.from_list(list(squad_task.validation_docs()))

    spec = TASK_SPECS[task]
    if spec["dataset_name"] is None:
        return load_dataset(
            spec["dataset_path"],
            split=spec["split"],
            trust_remote_code=True,
        )
    return load_dataset(
        spec["dataset_path"],
        spec["dataset_name"],
        split=spec["split"],
        trust_remote_code=True,
    )


def generate_with_lm_eval(docs, args) -> tuple[list[str], list[str]]:
    lm = get_model(args.model).create_from_arg_obj(
        build_model_args(args),
        additional_config={"batch_size": "auto"},
    )
    gen_kwargs = {
        "temperature": 0.7,
        "top_p": 0.95,
        "do_sample": True,
        "max_gen_toks": args.max_gen_toks,
        "until": ["<turn|>", "<|turn>"],
        "skip_special_tokens": False,
    }
    prompts = build_stage1_chat_prompts(lm, args.task, docs)
    requests = [
        Instance(
            request_type="generate_until",
            doc=doc,
            arguments=(prompt, gen_kwargs),
            idx=0,
            metadata=(f"{args.task}_cot_trace", i, 1),
        )
        for i, (doc, prompt) in enumerate(zip(docs, prompts))
    ]
    return [text.strip() for text in lm.generate_until(requests)], prompts


def build_stage1_chat_prompts(lm, task: str, docs: list[dict]) -> list[str]:
    return [
        lm.apply_chat_template(
            [{"role": "user", "content": build_user_prompt(task, doc)}],
            add_generation_prompt=True,
        )
        for doc in docs
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=sorted(TASK_SPECS))
    parser.add_argument("--model", default="vllm")
    parser.add_argument("--model-args", required=True)
    parser.add_argument("--output")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-gen-toks", type=int, default=65536)
    args = parser.parse_args()

    ds = load_task_dataset(args.task)
    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))

    docs = list(ds)
    traces, chat_prompts = generate_with_lm_eval(docs, args)
    ds = ds.add_column("stage1_chat_prompt", chat_prompts)
    ds = ds.add_column("reasoning_trace", traces)
    output = Path(args.output or TASK_SPECS[args.task]["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(output))
    print(f"Wrote {len(ds)} traces to {output}")


if __name__ == "__main__":
    main()
