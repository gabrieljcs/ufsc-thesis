from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from thesis_eval.cli import main
from thesis_eval.io import read_jsonl, write_jsonl
from thesis_eval.benchmarks.belebele import compute_if_cons
from thesis_eval.benchmarks.belebele_likelihood import (
    attach_option_likelihood_predictions,
    build_option_continuations,
    select_choice,
)


def _prediction_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "model": "llamantino_2_ultrachat_7b",
        "language": "ita",
        "item_id": "belebele_0001",
        "canonical_item_id": "belebele:giza",
        "gold": "C",
        "choice_a": "Tombe",
        "choice_b": "Templi",
        "choice_c": "Tutte le sette meraviglie",
        "choice_d": "La Grande Piramide",
    }
    row.update(overrides)
    return row


def _stats(pairs: dict[str, tuple[float, int]]) -> dict[str, dict[str, float | int]]:
    return {
        label: {"logprob_sum": total, "token_count": count}
        for label, (total, count) in pairs.items()
    }


class BuildOptionContinuationsTests(unittest.TestCase):
    def test_option_text_mode_returns_four_labelled_option_texts_in_order(self) -> None:
        continuations = build_option_continuations(_prediction_row(), mode="option_text")

        self.assertEqual(
            continuations,
            [
                ("A", " Tombe"),
                ("B", " Templi"),
                ("C", " Tutte le sette meraviglie"),
                ("D", " La Grande Piramide"),
            ],
        )

    def test_letter_mode_returns_bare_option_labels(self) -> None:
        continuations = build_option_continuations(_prediction_row(), mode="letter")

        self.assertEqual(continuations, [("A", " A"), ("B", " B"), ("C", " C"), ("D", " D")])

    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_option_continuations(_prediction_row(), mode="embedding")


class SelectChoiceTests(unittest.TestCase):
    def test_mean_normalization_prefers_higher_per_token_logprob_over_higher_total(self) -> None:
        # A is the best total (-3.0 > -4.0) but the worst per token (-3.0 < -1.0).
        # Unnormalized scoring would pick the shortest option regardless of fit.
        stats = _stats({"A": (-3.0, 1), "B": (-4.0, 4), "C": (-9.0, 3), "D": (-20.0, 5)})

        self.assertEqual(select_choice(stats, normalization="mean"), "B")

    def test_sum_normalization_prefers_higher_total_logprob(self) -> None:
        stats = _stats({"A": (-3.0, 1), "B": (-4.0, 4), "C": (-9.0, 3), "D": (-20.0, 5)})

        self.assertEqual(select_choice(stats, normalization="sum"), "A")

    def test_zero_token_continuation_is_rejected_rather_than_dividing_by_zero(self) -> None:
        stats = _stats({"A": (0.0, 0), "B": (-4.0, 4), "C": (-9.0, 3), "D": (-20.0, 5)})

        with self.assertRaises(ValueError):
            select_choice(stats, normalization="mean")

    def test_zero_token_continuation_is_rejected_under_sum_normalization_too(self) -> None:
        # Log-probabilities are negative, so a zero-token option scores 0.0 and
        # would always win an unnormalized comparison.
        stats = _stats({"A": (0.0, 0), "B": (-4.0, 4), "C": (-9.0, 3), "D": (-20.0, 5)})

        with self.assertRaises(ValueError):
            select_choice(stats, normalization="sum")

    def test_missing_option_is_rejected(self) -> None:
        stats = _stats({"A": (-3.0, 1), "B": (-4.0, 4), "C": (-9.0, 3)})

        with self.assertRaises(ValueError):
            select_choice(stats, normalization="mean")


class AttachOptionLikelihoodPredictionsTests(unittest.TestCase):
    def test_every_row_receives_a_choice_so_parse_coverage_is_total(self) -> None:
        rows = [_prediction_row(), _prediction_row(canonical_item_id="belebele:skype")]
        scores = [
            _stats({"A": (-9.0, 2), "B": (-9.0, 2), "C": (-1.0, 4), "D": (-9.0, 3)}),
            _stats({"A": (-1.0, 2), "B": (-9.0, 2), "C": (-9.0, 4), "D": (-9.0, 3)}),
        ]

        enriched = attach_option_likelihood_predictions(
            rows, scores, normalization="mean", mode="option_text"
        )

        self.assertEqual([row["predicted_choice"] for row in enriched], ["C", "A"])
        self.assertTrue(all(row["predicted_choice"] is not None for row in enriched))

    def test_both_normalizations_and_token_counts_are_persisted_for_offline_recomputation(self) -> None:
        stats = _stats({"A": (-3.0, 1), "B": (-4.0, 4), "C": (-9.0, 3), "D": (-20.0, 5)})

        enriched = attach_option_likelihood_predictions(
            [_prediction_row()], [stats], normalization="mean", mode="option_text"
        )[0]

        self.assertEqual(enriched["scoring_method"], "option_likelihood")
        self.assertEqual(enriched["likelihood_normalization"], "mean")
        self.assertEqual(enriched["continuation_mode"], "option_text")
        self.assertEqual(enriched["option_token_counts"], {"A": 1, "B": 4, "C": 3, "D": 5})
        self.assertEqual(
            enriched["option_logprob_sum"], {"A": -3.0, "B": -4.0, "C": -9.0, "D": -20.0}
        )
        self.assertEqual(enriched["option_logprob_mean"]["B"], -1.0)
        self.assertEqual(enriched["predicted_choice_mean"], "B")
        self.assertEqual(enriched["predicted_choice_sum"], "A")
        self.assertEqual(enriched["predicted_choice"], "B")

    def test_row_and_score_count_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            attach_option_likelihood_predictions(
                [_prediction_row()], [], normalization="mean", mode="option_text"
            )


class LikelihoodRowsFeedSpecComputationTests(unittest.TestCase):
    def test_if_cons_sees_full_parse_and_comparison_coverage(self) -> None:
        aligned = {"llamantino_2_ultrachat_7b": "ita"}
        correct = _stats({"A": (-9.0, 2), "B": (-9.0, 2), "C": (-1.0, 4), "D": (-9.0, 3)})
        wrong = _stats({"A": (-1.0, 2), "B": (-9.0, 2), "C": (-9.0, 4), "D": (-9.0, 3)})
        rows = attach_option_likelihood_predictions(
            [
                _prediction_row(language="ita"),
                _prediction_row(language="rus"),
            ],
            [correct, wrong],
            normalization="mean",
            mode="option_text",
        )

        scored = {row["attack_language"]: row for row in compute_if_cons(rows, aligned)}

        self.assertEqual(scored["ita"]["n_parseable"], 1)
        self.assertEqual(scored["ita"]["if_score"], 1.0)
        self.assertEqual(scored["rus"]["n_parseable"], 1)
        self.assertEqual(scored["rus"]["if_score"], 0.0)
        # The aligned-language answer is always available, so every item is comparable.
        self.assertEqual(scored["rus"]["n_cons_comparable"], 1)
        self.assertEqual(scored["rus"]["cons_score"], 0.0)


class CompareScoringPathsTests(unittest.TestCase):
    def _generated(self, item: str, choice: str | None) -> dict[str, object]:
        return _prediction_row(
            canonical_item_id=item,
            predicted_choice=choice,
            prediction=choice if choice is not None else "La risposta cor",
            scoring_method="generated_parse",
        )

    def _likelihood(self, item: str, choice: str) -> dict[str, object]:
        return _prediction_row(
            canonical_item_id=item,
            predicted_choice=choice,
            prediction=choice,
            scoring_method="option_likelihood",
        )

    def test_agreement_is_measured_only_on_rows_the_generation_path_parsed(self) -> None:
        generated = [self._generated("i1", "C"), self._generated("i2", None)]
        likelihood = [self._likelihood("i1", "C"), self._likelihood("i2", "A")]

        from thesis_eval.benchmarks.belebele_likelihood import compare_scoring_paths

        summary = compare_scoring_paths(generated, likelihood)[0]

        self.assertEqual(summary["n_joined"], 2)
        self.assertEqual(summary["n_generated_parsed"], 1)
        self.assertEqual(summary["n_agreed"], 1)
        self.assertEqual(summary["agreement_on_generated_parsed"], 1.0)

    def test_disagreement_lowers_the_agreement_rate(self) -> None:
        generated = [self._generated("i1", "C"), self._generated("i2", "B")]
        likelihood = [self._likelihood("i1", "C"), self._likelihood("i2", "A")]

        from thesis_eval.benchmarks.belebele_likelihood import compare_scoring_paths

        summary = compare_scoring_paths(generated, likelihood)[0]

        self.assertEqual(summary["agreement_on_generated_parsed"], 0.5)

    def test_if_is_reported_for_both_paths_so_the_shift_is_visible(self) -> None:
        # gold is C. Generation parses one correct answer and fails the other;
        # likelihood answers both, one correct.
        generated = [self._generated("i1", "C"), self._generated("i2", None)]
        likelihood = [self._likelihood("i1", "C"), self._likelihood("i2", "A")]

        from thesis_eval.benchmarks.belebele_likelihood import compare_scoring_paths

        summary = compare_scoring_paths(generated, likelihood)[0]

        self.assertEqual(summary["if_generated"], 0.5)
        self.assertEqual(summary["if_likelihood"], 0.5)
        self.assertEqual(summary["parse_rate_generated"], 0.5)
        self.assertEqual(summary["parse_rate_likelihood"], 1.0)

    def test_agreement_is_undefined_when_the_generation_path_parsed_nothing(self) -> None:
        generated = [self._generated("i1", None)]
        likelihood = [self._likelihood("i1", "A")]

        from thesis_eval.benchmarks.belebele_likelihood import compare_scoring_paths

        summary = compare_scoring_paths(generated, likelihood)[0]

        self.assertEqual(summary["n_generated_parsed"], 0)
        self.assertIsNone(summary["agreement_on_generated_parsed"])


def _write_belebele_fixture(data_dir: Path) -> None:
    data_dir.mkdir(parents=True)
    write_jsonl(
        data_dir / "por_Latn.jsonl",
        [
            {
                "question_number": 1,
                "link": "https://example.test/passage",
                "flores_passage": "Trecho",
                "question": "Pergunta?",
                "mc_answer1": "A1",
                "mc_answer2": "A2",
                "mc_answer3": "A3",
                "mc_answer4": "A4",
                "correct_answer_num": "2",
                "dialect": "por_Latn",
            }
        ],
    )


class PredictBelebeleLikelihoodCliTests(unittest.TestCase):
    def test_mock_backend_records_all_four_options_and_a_choice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_belebele_fixture(tmp_path / "belebele" / "data")
            output = tmp_path / "likelihood.jsonl"

            main(
                [
                    "predict-belebele-likelihood",
                    "--dataset-dir",
                    str(tmp_path / "belebele"),
                    "--language",
                    "por",
                    "--model",
                    "sagui_7b",
                    "--backend",
                    "mock",
                    "--output",
                    str(output),
                ]
            )

            rows = read_jsonl(output)

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["scoring_method"], "option_likelihood")
        self.assertEqual(sorted(row["option_logprob_sum"]), ["A", "B", "C", "D"])
        self.assertEqual(sorted(row["option_token_counts"]), ["A", "B", "C", "D"])
        self.assertIsNotNone(row["predicted_choice"])
        self.assertEqual(row["gold"], "B")
        self.assertEqual(row["language"], "por")

    def test_api_only_model_is_refused_because_logprobs_are_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_belebele_fixture(tmp_path / "belebele" / "data")

            with self.assertRaises(ValueError) as caught:
                main(
                    [
                        "predict-belebele-likelihood",
                        "--dataset-dir",
                        str(tmp_path / "belebele"),
                        "--language",
                        "por",
                        "--model",
                        "sabia_3",
                        "--backend",
                        "mock",
                        "--output",
                        str(tmp_path / "unused.jsonl"),
                    ]
                )

        self.assertIn("sabia_3", str(caught.exception))


class CompareBelebeleScoringCliTests(unittest.TestCase):
    def test_writes_one_agreement_row_per_model_language_cell(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            generated = tmp_path / "generated.jsonl"
            likelihood = tmp_path / "likelihood.jsonl"
            output = tmp_path / "agreement.jsonl"
            write_jsonl(
                generated,
                [
                    _prediction_row(canonical_item_id="i1", predicted_choice="C", prediction="C"),
                    _prediction_row(
                        canonical_item_id="i2",
                        predicted_choice=None,
                        prediction="La risposta cor",
                    ),
                ],
            )
            write_jsonl(
                likelihood,
                [
                    _prediction_row(canonical_item_id="i1", predicted_choice="C", prediction="C"),
                    _prediction_row(canonical_item_id="i2", predicted_choice="A", prediction="A"),
                ],
            )

            main(
                [
                    "compare-belebele-scoring",
                    "--generated",
                    str(generated),
                    "--likelihood",
                    str(likelihood),
                    "--output",
                    str(output),
                ]
            )

            rows = read_jsonl(output)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["n_joined"], 2)
        self.assertEqual(rows[0]["agreement_on_generated_parsed"], 1.0)
        self.assertEqual(rows[0]["parse_rate_likelihood"], 1.0)


if __name__ == "__main__":
    unittest.main()
