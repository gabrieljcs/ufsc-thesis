import unittest

import numpy as np
import pandas as pd

from thesis_eval.analysis.magnitude import (
    cluster_groups,
    combine_twoway_covariance,
    fit_cluster_variants,
)


def _cell_level_panel(seed: int = 11) -> pd.DataFrame:
    """A predictor that varies only between model--language cells.

    Each cell also carries its own shock, so residuals correlate within a cell
    but not within a prompt. Prompt clusters cut across every cell, so they
    cannot see that correlation.
    """
    rng = np.random.default_rng(seed)
    cells = {}
    for model in ("a", "b"):
        for language in ("a", "b", "c"):
            cells[(model, language)] = (rng.normal(), rng.normal() * 1.5)
    rows = []
    for prompt in range(120):
        for (model, language), (x, shock) in cells.items():
            linear = 0.4 * x + shock + rng.normal(scale=0.3)
            score = 1.0 / (1.0 + np.exp(-linear))
            rows.append(
                dict(
                    prompt_id=str(prompt),
                    model=model,
                    attack_language=language,
                    x=x,
                    strongreject_score=float(score),
                )
            )
    return pd.DataFrame(rows)


class ClusterGroupsTests(unittest.TestCase):
    def test_prompt_clustering_uses_the_prompt_identifier(self) -> None:
        frame = pd.DataFrame(
            dict(prompt_id=["a", "b"], model=["m", "m"], attack_language=["x", "y"])
        )
        self.assertEqual(list(cluster_groups(frame, "prompt_id")), ["a", "b"])

    def test_cell_clustering_pairs_model_with_attack_language(self) -> None:
        frame = pd.DataFrame(
            dict(prompt_id=["a", "b"], model=["m", "m"], attack_language=["x", "y"])
        )
        self.assertEqual(list(cluster_groups(frame, "cell")), ["m|x", "m|y"])

    def test_cell_clustering_collapses_prompts_within_one_cell(self) -> None:
        frame = pd.DataFrame(
            dict(prompt_id=["a", "b"], model=["m", "m"], attack_language=["x", "x"])
        )
        self.assertEqual(len(set(cluster_groups(frame, "cell"))), 1)

    def test_unknown_clustering_is_rejected(self) -> None:
        frame = pd.DataFrame(dict(prompt_id=["a"], model=["m"], attack_language=["x"]))
        with self.assertRaises(ValueError):
            cluster_groups(frame, "country")


class CombineTwowayCovarianceTests(unittest.TestCase):
    def test_adds_the_two_dimensions_and_subtracts_their_intersection(self) -> None:
        a = np.array([[4.0, 0.0], [0.0, 4.0]])
        b = np.array([[9.0, 0.0], [0.0, 9.0]])
        intersection = np.array([[1.0, 0.0], [0.0, 1.0]])

        combined, positive_definite = combine_twoway_covariance(a, b, intersection)

        np.testing.assert_allclose(combined, [[12.0, 0.0], [0.0, 12.0]])
        self.assertTrue(positive_definite)

    def test_reports_when_the_combination_is_not_positive_definite(self) -> None:
        a = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([[1.0, 0.0], [0.0, 1.0]])
        intersection = np.array([[5.0, 0.0], [0.0, 5.0]])

        _, positive_definite = combine_twoway_covariance(a, b, intersection)

        self.assertFalse(positive_definite)


class FitClusterVariantsTests(unittest.TestCase):
    def test_point_estimates_are_identical_across_clusterings(self) -> None:
        # Working independence means the grouping enters only the sandwich
        # covariance, never the estimating equations. If the coefficients moved,
        # the comparison would not be a pure inference comparison.
        variants = fit_cluster_variants(_cell_level_panel(), ("x",), "synthetic")

        estimates = {v["cluster"]: v["estimate"] for v in variants}
        self.assertAlmostEqual(estimates["prompt_id"], estimates["cell"], places=10)
        self.assertAlmostEqual(estimates["prompt_id"], estimates["twoway"], places=10)

    def test_cell_clustering_widens_the_interval_for_a_cell_level_predictor(self) -> None:
        variants = {
            v["cluster"]: v for v in fit_cluster_variants(_cell_level_panel(), ("x",), "synthetic")
        }

        self.assertGreater(
            variants["cell"]["std_error"], variants["prompt_id"]["std_error"]
        )

    def test_each_variant_reports_its_cluster_count(self) -> None:
        variants = {
            v["cluster"]: v for v in fit_cluster_variants(_cell_level_panel(), ("x",), "synthetic")
        }

        self.assertEqual(variants["prompt_id"]["n_clusters"], 120)
        self.assertEqual(variants["cell"]["n_clusters"], 6)


if __name__ == "__main__":
    unittest.main()
