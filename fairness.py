import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

FAIRNESS_BENCHMARK_METRICS = {
    "StereoSet-UK Eval": ("LMS ↑", "SS → 50", "ICAT ↑"),
    "WinoBias-UK": (
        "Type 1 F1 ↑",
        "Type 1 pro/anti gap → 0",
        "Type 2 F1 ↑",
        "Type 2 pro/anti gap → 0",
    ),
    "WinoGender-UK": (
        "Accuracy ↑",
        "Gender accuracy gap → 0",
    ),
    "BBQ-UK": (
        "Ambiguous accuracy ↑",
        "Disambiguated accuracy ↑",
        "Ambiguous bias → 0",
        "Disambiguated bias → 0",
    ),
    "CrowS-Pairs-UK": ("Stereotype score → 50",),
}


@dataclass(frozen=True)
class FairnessRanking:
    metric: str
    goal: str
    target: float | None = None


@dataclass
class FairnessBenchmark:
    ranking: FairnessRanking
    overall_rows: list[dict[str, str | int | float]]
    bias_type_rows: list[dict[str, str | int | float]]


def create_ranked_fairness_dataframe(
    rows: list[dict[str, str | int | float]],
    ranking: FairnessRanking,
    group_column: str | None = None,
) -> pd.DataFrame:
    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        return dataframe

    ranking_values = dataframe[ranking.metric]
    ascending = ranking.goal != "maximize"
    if ranking.goal == "target":
        ranking_values = (ranking_values - ranking.target).abs()

    if group_column:
        ranks = ranking_values.groupby(dataframe[group_column]).rank(
            method="min", ascending=ascending
        )
        dataframe.insert(1, "Rank", ranks.astype(int))
        return dataframe.sort_values([group_column, "Rank", "Model"])

    ranks = ranking_values.rank(method="min", ascending=ascending)
    dataframe.insert(0, "Rank", ranks.astype(int))
    return dataframe.sort_values(["Rank", "Model"])


def create_bias_type_scorecard(
    ranked_dataframe: pd.DataFrame,
    ranking: FairnessRanking,
) -> pd.DataFrame:
    bias_types = ranked_dataframe["Bias type"].drop_duplicates().tolist()
    scores = ranked_dataframe.pivot(
        index="Model", columns="Bias type", values=ranking.metric
    ).reindex(columns=bias_types)
    ranks = ranked_dataframe.pivot(
        index="Model", columns="Bias type", values="Rank"
    ).reindex(columns=bias_types)

    scorecard = scores.map(lambda score: "—" if pd.isna(score) else f"{score:.2f}")
    for bias_type in bias_types:
        winners = ranks[bias_type] == 1
        scorecard.loc[winners, bias_type] = "★ " + scorecard.loc[winners, bias_type]

    scorecard.insert(0, "Wins", (ranks == 1).sum(axis=1))
    return scorecard.reset_index().sort_values(["Wins", "Model"], ascending=[False, True])


def _load_scores(path: Path, label: str, scores: dict[str, Any]) -> dict[str, float]:
    loaded_scores = {}
    for metric_name, metric_value in scores.items():
        value = float(metric_value)
        if not math.isfinite(value):
            raise ValueError(f"{path}: {label} · {metric_name} must be finite")
        loaded_scores[metric_name] = round(value, 2)
    return loaded_scores


def _load_ranking(path: Path, benchmark: dict[str, Any]) -> FairnessRanking:
    ranking = benchmark["ranking"]
    goal = ranking["goal"]
    if goal not in {"maximize", "minimize", "target"}:
        raise ValueError(f"{path}: unsupported ranking goal {goal}")
    target = float(ranking["target"]) if goal == "target" else None
    return FairnessRanking(metric=ranking["metric"], goal=goal, target=target)


def _load_result(
    path: Path,
) -> tuple[str, list[tuple[str, FairnessRanking, dict, list[dict]]]]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    model_name = payload["model"]["name"]
    loaded_benchmarks = []
    benchmark_names = set()

    for benchmark in payload["benchmarks"]:
        benchmark_name = benchmark["name"]
        if benchmark_name in benchmark_names:
            raise ValueError(f"{path}: duplicate benchmark {benchmark_name}")
        benchmark_names.add(benchmark_name)

        ranking = _load_ranking(path, benchmark)
        scores = _load_scores(path, benchmark_name, benchmark["scores"])
        if ranking.metric not in scores:
            raise ValueError(f"{path}: missing ranking metric {ranking.metric}")

        overall_row = {"Model": model_name, **scores}
        bias_type_rows = []
        for bias_type in benchmark.get("bias_types", []):
            bias_type_name = bias_type["name"]
            bias_type_scores = _load_scores(path, bias_type_name, bias_type["scores"])
            if ranking.metric not in bias_type_scores:
                raise ValueError(f"{path}: {bias_type_name} missing {ranking.metric}")
            bias_type_rows.append(
                {
                    "Bias type": bias_type_name,
                    "Items": int(bias_type["items"]),
                    "Model": model_name,
                    **bias_type_scores,
                }
            )

        loaded_benchmarks.append(
            (benchmark_name, ranking, overall_row, bias_type_rows)
        )

    return model_name, loaded_benchmarks


def load_fairness_benchmarks(
    results_dir: str | Path = "eval-results/fairness",
) -> dict[str, FairnessBenchmark]:
    results = [_load_result(path) for path in sorted(Path(results_dir).glob("*.json"))]
    model_counts = Counter(model_name for model_name, _ in results)
    duplicate_models = sorted(model for model, count in model_counts.items() if count > 1)
    if duplicate_models:
        raise ValueError(f"Duplicate fairness results for: {', '.join(duplicate_models)}")

    benchmarks = {}
    for _, model_benchmarks in results:
        for benchmark_name, ranking, overall_row, bias_type_rows in model_benchmarks:
            benchmark = benchmarks.setdefault(
                benchmark_name,
                FairnessBenchmark(ranking=ranking, overall_rows=[], bias_type_rows=[]),
            )
            if benchmark.ranking != ranking:
                raise ValueError(f"Inconsistent ranking for {benchmark_name}")
            benchmark.overall_rows.append(overall_row)
            benchmark.bias_type_rows.extend(bias_type_rows)

    return benchmarks


def render_fairness_tab() -> None:
    benchmarks = load_fairness_benchmarks()
    benchmark_names = list(dict.fromkeys([*FAIRNESS_BENCHMARK_METRICS, *benchmarks]))

    with gr.Tab("⚖️ Fairness"):
        gr.Markdown(
            """
        ## Ukrainian Fairness Leaderboard

        Each benchmark has its own table. Metric arrows show whether higher values, zero, or 50 are preferred.
        """
        )
        with gr.Tabs():
            for benchmark_name in benchmark_names:
                benchmark = benchmarks.get(benchmark_name)
                metrics = FAIRNESS_BENCHMARK_METRICS.get(benchmark_name, ())
                with gr.Tab(benchmark_name):
                    if benchmark_name == "StereoSet-UK Eval":
                        gr.Markdown(
                            """
                        **LMS ↑** measures preference for related completions. **SS → 50** measures stereotype preference, with 50 as the neutral point. **ICAT ↑** combines language-model quality and stereotype neutrality.
                        """
                        )
                        if benchmark is not None:
                            gr.Markdown(
                                "Current results cover the provisional 949-item [StereoSet-UK Eval](https://huggingface.co/datasets/FairForget/StereoSet-UK-Eval) subset."
                            )
                    if benchmark is None:
                        gr.Markdown("Results pending. Planned metrics appear below.")
                        gr.Dataframe(
                            value=pd.DataFrame(columns=["Model", *metrics]),
                            label=benchmark_name,
                            interactive=False,
                            wrap=False,
                        )
                    elif benchmark.bias_type_rows:
                        bias_type_dataframe = create_ranked_fairness_dataframe(
                            benchmark.bias_type_rows,
                            benchmark.ranking,
                            "Bias type",
                        )
                        with gr.Tabs():
                            with gr.Tab("Overall"):
                                gr.Dataframe(
                                    value=create_ranked_fairness_dataframe(
                                        benchmark.overall_rows,
                                        benchmark.ranking,
                                    ),
                                    label=f"{benchmark_name} overall",
                                    interactive=False,
                                    wrap=False,
                                )
                            with gr.Tab("By bias type"):
                                gr.Markdown(
                                    f"Each column is a bias type. ★ marks the best **{benchmark.ranking.metric}** score."
                                )
                                gr.Dataframe(
                                    value=create_bias_type_scorecard(
                                        bias_type_dataframe,
                                        benchmark.ranking,
                                    ),
                                    label="Bias type comparison",
                                    interactive=False,
                                    wrap=False,
                                )
                    else:
                        gr.Dataframe(
                            value=create_ranked_fairness_dataframe(
                                benchmark.overall_rows,
                                benchmark.ranking,
                            ),
                            label=benchmark_name,
                            interactive=False,
                            wrap=False,
                        )
