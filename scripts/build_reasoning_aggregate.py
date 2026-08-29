#!/usr/bin/env python3
import argparse
import copy
import json
from pathlib import Path
from typing import Optional


CANONICAL_TASKS = {
    "arc_easy_uk_cot_likelihood": "arc_easy_uk",
    "winogrande_uk_cot_likelihood": "winogrande_uk",
    "belebele_ukr_Cyrl_cot_likelihood": "belebele_ukr_Cyrl",
    "global_mmlu_full_uk_cot_likelihood": "global_mmlu_full_uk",
    "squad_uk_cot_likelihood": "squad_uk",
}

METRIC_NORMALIZATION = {
    "arc_challenge_uk": {
        "exact_match,extract_answer": "exact_match,none",
        "exact_match_stderr,extract_answer": "exact_match_stderr,none",
    },
    "wmt_en_uk": {
        "bleu,extract_translation": "bleu,none",
        "bleu_stderr,extract_translation": "bleu_stderr,none",
    },
    "flores_en-uk": {
        "bleu,extract_translation": "bleu,none",
        "bleu_stderr,extract_translation": "bleu_stderr,none",
    },
    "flores_uk-en": {
        "bleu,extract_translation": "bleu,none",
        "bleu_stderr,extract_translation": "bleu_stderr,none",
    },
    "long_flores_en-uk": {
        "bleu,extract_translation": "bleu,none",
        "bleu_stderr,extract_translation": "bleu_stderr,none",
    },
    "long_flores_uk-en": {
        "bleu,extract_translation": "bleu,none",
        "bleu_stderr,extract_translation": "bleu_stderr,none",
    },
    "zno_uk_geography": {
        "exact,strict_letter": "exact,none",
        "exact_stderr,strict_letter": "exact_stderr,none",
    },
    "zno_uk_history": {
        "exact,strict_letter": "exact,none",
        "exact_stderr,strict_letter": "exact_stderr,none",
    },
    "zno_uk_language_and_literature": {
        "exact,strict_letter": "exact,none",
        "exact_stderr,strict_letter": "exact_stderr,none",
    },
    "zno_uk_math": {
        "exact,strict_letter": "exact,none",
        "exact_stderr,strict_letter": "exact_stderr,none",
    },
}


def read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def metric_value(result: dict, metric: str) -> float:
    for key in (f"{metric},none", f"{metric},extract_translation", metric):
        if key in result:
            return result[key]
    raise KeyError(f"Missing {metric} in {result.get('alias')}")


def canonicalize_task(task: str) -> str:
    return CANONICAL_TASKS.get(task, task)


def normalize_metrics(task: str, result: dict) -> dict:
    result = copy.deepcopy(result)
    result["alias"] = task
    for src, dst in METRIC_NORMALIZATION.get(task, {}).items():
        if src in result:
            result[dst] = result[src]
    return result


def portable_result_path(path: Path) -> str:
    parts = path.parts
    for i, part in enumerate(parts):
        if part.startswith("eval-results"):
            return str(Path(*parts[i:]))
    return path.name


def reasoning_model_name(source_docs: list[dict], explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    for doc in source_docs:
        model_name = doc.get("model_name")
        if model_name:
            if "(reasoning)" in model_name:
                return model_name
            return f"{model_name} (reasoning)"
    return "reasoning-model"


def reasoning_model_name_sanitized(source_docs: list[dict], explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    for doc in source_docs:
        model_name = doc.get("model_name_sanitized")
        if model_name:
            if model_name.endswith("-reasoning"):
                return model_name
            return f"{model_name}-reasoning"
    return "reasoning-model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-name")
    parser.add_argument("--model-name-sanitized")
    parser.add_argument("source_results", nargs="+", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_docs = [read_json(path) for path in args.source_results]
    aggregate = copy.deepcopy(source_docs[-1])
    aggregate["results"] = {}
    aggregate["groups"] = {}
    aggregate["group_subtasks"] = {}
    aggregate["configs"] = {}
    aggregate["versions"] = {}
    aggregate["n-shot"] = {}
    aggregate["higher_is_better"] = {}
    aggregate["n-samples"] = {}

    for path, doc in zip(args.source_results, source_docs):
        for source_task, source_result in doc.get("results", {}).items():
            task = canonicalize_task(source_task)
            if task.endswith("_reasoning_api"):
                continue
            result = normalize_metrics(task, source_result)
            aggregate["results"][task] = result
            aggregate["group_subtasks"][task] = []

            config = copy.deepcopy(doc.get("configs", {}).get(source_task, {}))
            config["task"] = task
            config["alias"] = task
            config.setdefault("metadata", {})
            config["metadata"]["source_task"] = source_task
            config["metadata"]["source_result_file"] = portable_result_path(path)
            aggregate["configs"][task] = config

            aggregate["versions"][task] = doc.get("versions", {}).get(source_task, 0)
            aggregate["n-shot"][task] = doc.get("n-shot", {}).get(source_task, 0)
            aggregate["higher_is_better"][task] = doc.get(
                "higher_is_better", {}
            ).get(source_task, {})
            if source_task in doc.get("n-samples", {}):
                aggregate["n-samples"][task] = doc["n-samples"][source_task]

    if "flores_en-uk" in aggregate["results"] and "flores_uk-en" in aggregate["results"]:
        aggregate["results"]["flores_uk"] = {
            "alias": "flores_uk",
            "bleu,none": (
                metric_value(aggregate["results"]["flores_en-uk"], "bleu")
                + metric_value(aggregate["results"]["flores_uk-en"], "bleu")
            )
            / 2,
        }
        aggregate["n-shot"]["flores_uk"] = 0
        aggregate["higher_is_better"]["flores_uk"] = {"bleu": True}

    if (
        "long_flores_en-uk" in aggregate["results"]
        and "long_flores_uk-en" in aggregate["results"]
    ):
        aggregate["results"]["long_flores_uk"] = {
            "alias": "long_flores_uk",
            "bleu,none": (
                metric_value(aggregate["results"]["long_flores_en-uk"], "bleu")
                + metric_value(aggregate["results"]["long_flores_uk-en"], "bleu")
            )
            / 2,
        }
        aggregate["n-shot"]["long_flores_uk"] = 0
        aggregate["higher_is_better"]["long_flores_uk"] = {"bleu": True}

    aggregate["model_name"] = reasoning_model_name(source_docs, args.model_name)
    aggregate["model_name_sanitized"] = reasoning_model_name_sanitized(
        source_docs, args.model_name_sanitized
    )
    aggregate["model_source"] = "vllm"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
