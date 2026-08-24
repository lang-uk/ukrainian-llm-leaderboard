import json
import tempfile
import unittest
from pathlib import Path

from fairness import (
    FAIRNESS_BENCHMARK_DESCRIPTIONS,
    FAIRNESS_BENCHMARK_METRICS,
    FAIRNESS_BENCHMARK_RELEASE_NOTES,
    FairnessBenchmark,
    FairnessRanking,
    load_fairness_benchmarks,
)


def write_result(directory: Path, filename: str, model: str, benchmarks: list[dict]) -> None:
    payload = {"model": {"name": model, "revision": "revision"}, "benchmarks": benchmarks}
    (directory / filename).write_text(json.dumps(payload), encoding="utf-8")


def stereoset_result(icat: float = 70.0) -> dict:
    return {
        "name": "StereoSet-UK Eval",
        "ranking": {"metric": "ICAT ↑", "goal": "maximize"},
        "scores": {"LMS ↑": 80.0, "SS → 50": 50.0, "ICAT ↑": icat},
        "bias_types": [
            {
                "name": "Gender",
                "items": 10,
                "scores": {"LMS ↑": 60.0, "SS → 50": 40.0, "ICAT ↑": 55.0},
            }
        ],
    }


def winobias_result(worst_group_accuracy: float = 72.0) -> dict:
    return {
        "name": "WinoBias-UK Natural",
        "ranking": {"metric": "Worst-group accuracy ↑", "goal": "maximize"},
        "scores": {
            "Worst-group accuracy ↑": worst_group_accuracy,
            "Primary accuracy ↑": 78.0,
            "Pro/anti gap → 0": 12.0,
            "Pair consistency ↑": 68.0,
            "Agreement control ↑": 99.0,
            "Cross control ↑": 91.0,
            "Tie rate → 0": 0.0,
        },
    }


class FairnessResultsTests(unittest.TestCase):
    def test_loads_overall_and_bias_type_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            results_dir = Path(directory)
            write_result(results_dir, "model.json", "example/model", [stereoset_result()])

            benchmarks = load_fairness_benchmarks(results_dir)

        self.assertEqual(
            benchmarks,
            {
                "StereoSet-UK Eval": FairnessBenchmark(
                    ranking=FairnessRanking(metric="ICAT ↑", goal="maximize"),
                    overall_rows=[
                        {
                            "Model": "example/model",
                            "LMS ↑": 80.0,
                            "SS → 50": 50.0,
                            "ICAT ↑": 70.0,
                        }
                    ],
                    bias_type_rows=[
                        {
                            "Bias type": "Gender",
                            "Items": 10,
                            "Model": "example/model",
                            "LMS ↑": 60.0,
                            "SS → 50": 40.0,
                            "ICAT ↑": 55.0,
                        }
                    ],
                )
            },
        )

    def test_loads_each_benchmark_without_display_code_changes(self):
        bbq = {
            "name": "BBQ-UK",
            "ranking": {"metric": "Accuracy ↑", "goal": "maximize"},
            "scores": {"Accuracy ↑": 81.456},
        }
        with tempfile.TemporaryDirectory() as directory:
            results_dir = Path(directory)
            write_result(results_dir, "model.json", "example/model", [bbq])

            benchmarks = load_fairness_benchmarks(results_dir)

        self.assertEqual(
            benchmarks["BBQ-UK"].overall_rows,
            [{"Model": "example/model", "Accuracy ↑": 81.46}],
        )

    def test_loads_winobias_natural_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            results_dir = Path(directory)
            write_result(
                results_dir,
                "model.json",
                "example/model",
                [stereoset_result(), winobias_result()],
            )

            benchmarks = load_fairness_benchmarks(results_dir)

        winobias = benchmarks["WinoBias-UK Natural"]
        self.assertEqual(
            winobias.ranking,
            FairnessRanking(metric="Worst-group accuracy ↑", goal="maximize"),
        )
        self.assertEqual(winobias.overall_rows[0]["Pro/anti gap → 0"], 12.0)

    def test_rejects_duplicate_model_results(self):
        with tempfile.TemporaryDirectory() as directory:
            results_dir = Path(directory)
            write_result(results_dir, "first.json", "example/model", [stereoset_result()])
            write_result(results_dir, "second.json", "example/model", [stereoset_result()])

            with self.assertRaisesRegex(ValueError, "Duplicate fairness results"):
                load_fairness_benchmarks(results_dir)

    def test_rejects_duplicate_benchmarks(self):
        with tempfile.TemporaryDirectory() as directory:
            results_dir = Path(directory)
            benchmark = stereoset_result()
            write_result(results_dir, "model.json", "example/model", [benchmark, benchmark])

            with self.assertRaisesRegex(ValueError, "duplicate benchmark"):
                load_fairness_benchmarks(results_dir)

    def test_rejects_missing_ranking_metric(self):
        with tempfile.TemporaryDirectory() as directory:
            results_dir = Path(directory)
            benchmark = stereoset_result()
            benchmark["ranking"]["metric"] = "Missing"
            write_result(results_dir, "model.json", "example/model", [benchmark])

            with self.assertRaisesRegex(ValueError, "missing ranking metric"):
                load_fairness_benchmarks(results_dir)

    def test_rejects_non_finite_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            results_dir = Path(directory)
            write_result(results_dir, "model.json", "example/model", [stereoset_result(float("nan"))])

            with self.assertRaisesRegex(ValueError, "must be finite"):
                load_fairness_benchmarks(results_dir)

    def test_returns_no_benchmarks_when_no_results_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(load_fairness_benchmarks(directory), {})

    def test_benchmark_stubs_have_metric_headers(self):
        self.assertEqual(
            set(FAIRNESS_BENCHMARK_METRICS),
            {
                "StereoSet-UK Eval",
                "WinoBias-UK Natural",
                "WinoPron-UK",
                "BBQ-UK",
            },
        )
        self.assertTrue(all(FAIRNESS_BENCHMARK_METRICS.values()))
        self.assertEqual(
            set(FAIRNESS_BENCHMARK_DESCRIPTIONS),
            set(FAIRNESS_BENCHMARK_METRICS),
        )
        self.assertEqual(
            set(FAIRNESS_BENCHMARK_RELEASE_NOTES),
            set(FAIRNESS_BENCHMARK_METRICS),
        )
        for benchmark_name, metrics in FAIRNESS_BENCHMARK_METRICS.items():
            description = FAIRNESS_BENCHMARK_DESCRIPTIONS[benchmark_name]
            for metric in metrics:
                self.assertIn(metric, description)


if __name__ == "__main__":
    unittest.main()
