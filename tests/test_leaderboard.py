import unittest

from fairness import (
    FairnessRanking,
    create_bias_type_scorecard,
    create_ranked_fairness_dataframe,
)


class LeaderboardResultsTests(unittest.TestCase):
    def test_ranks_models_within_each_bias_type(self):
        rows = [
            {"Bias type": "Gender", "Model": "model/a", "ICAT ↑": 50.0},
            {"Bias type": "Gender", "Model": "model/b", "ICAT ↑": 60.0},
            {"Bias type": "Religion", "Model": "model/a", "ICAT ↑": 70.0},
            {"Bias type": "Religion", "Model": "model/b", "ICAT ↑": 65.0},
        ]

        dataframe = create_ranked_fairness_dataframe(
            rows,
            FairnessRanking(metric="ICAT ↑", goal="maximize"),
            "Bias type",
        )

        self.assertEqual(
            dataframe[["Bias type", "Rank", "Model"]].to_dict("records"),
            [
                {"Bias type": "Gender", "Rank": 1, "Model": "model/b"},
                {"Bias type": "Gender", "Rank": 2, "Model": "model/a"},
                {"Bias type": "Religion", "Rank": 1, "Model": "model/a"},
                {"Bias type": "Religion", "Rank": 2, "Model": "model/b"},
            ],
        )

        scorecard = create_bias_type_scorecard(
            dataframe,
            FairnessRanking(metric="ICAT ↑", goal="maximize"),
        )
        self.assertEqual(
            scorecard.to_dict("records"),
            [
                {
                    "Model": "model/a",
                    "Wins": 1,
                    "Gender": "50.00",
                    "Religion": "★ 70.00",
                },
                {
                    "Model": "model/b",
                    "Wins": 1,
                    "Gender": "★ 60.00",
                    "Religion": "65.00",
                },
            ],
        )

    def test_ranks_target_metrics_by_distance(self):
        rows = [
            {"Model": "model/a", "Stereotype score → 50": 60.0},
            {"Model": "model/b", "Stereotype score → 50": 52.0},
        ]

        dataframe = create_ranked_fairness_dataframe(
            rows,
            FairnessRanking(
                metric="Stereotype score → 50", goal="target", target=50.0
            ),
        )

        self.assertEqual(dataframe["Model"].tolist(), ["model/b", "model/a"])

if __name__ == "__main__":
    unittest.main()
