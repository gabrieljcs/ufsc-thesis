from __future__ import annotations

import math
import unittest

import numpy as np

from thesis_eval.models.likelihood import (
    common_prefix_length,
    continuation_logit_positions,
    plan_sequence_batches,
    score_continuation,
)


class PlanSequenceBatchesTests(unittest.TestCase):
    def test_a_batch_spans_several_items_so_the_gpu_is_not_underfed(self) -> None:
        # Three items of four options each, batched eight at a time: the first
        # batch must cover items 0 and 1 rather than stopping at one item.
        batches = plan_sequence_batches([4, 4, 4], batch_size=8)

        self.assertEqual(len(batches), 2)
        self.assertEqual(
            batches[0],
            [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)],
        )
        self.assertEqual(batches[1], [(2, 0), (2, 1), (2, 2), (2, 3)])

    def test_every_sequence_is_scheduled_exactly_once(self) -> None:
        counts = [4, 4, 4, 4, 4]
        scheduled = [pair for batch in plan_sequence_batches(counts, batch_size=6) for pair in batch]

        expected = [(item, seq) for item, count in enumerate(counts) for seq in range(count)]
        self.assertEqual(scheduled, expected)

    def test_batch_smaller_than_one_item_splits_within_the_item(self) -> None:
        batches = plan_sequence_batches([4], batch_size=3)

        self.assertEqual(batches, [[(0, 0), (0, 1), (0, 2)], [(0, 3)]])

    def test_no_items_yields_no_batches(self) -> None:
        self.assertEqual(plan_sequence_batches([], batch_size=8), [])

    def test_non_positive_batch_size_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            plan_sequence_batches([4], batch_size=0)


class CommonPrefixLengthTests(unittest.TestCase):
    def test_returns_the_shared_leading_token_count(self) -> None:
        self.assertEqual(common_prefix_length([1, 2, 3], [1, 2, 3, 9, 8]), 3)

    def test_detects_retokenization_at_the_continuation_boundary(self) -> None:
        # Joining a continuation can re-tokenize the last prefix token, so the
        # prefix length must be measured against the joint encoding, not assumed.
        self.assertEqual(common_prefix_length([1, 2, 30], [1, 2, 31, 7]), 2)

    def test_no_shared_prefix_returns_zero(self) -> None:
        self.assertEqual(common_prefix_length([5], [6, 5]), 0)


class ContinuationLogitPositionsTests(unittest.TestCase):
    def test_positions_are_shifted_one_left_of_the_continuation_tokens(self) -> None:
        # Causal LMs predict token t from the logits at position t-1. The
        # continuation occupies positions 2..3, so its predictions live at 1..2.
        self.assertEqual(continuation_logit_positions(prefix_len=2, total_len=4), [1, 2])

    def test_single_continuation_token_yields_a_single_position(self) -> None:
        self.assertEqual(continuation_logit_positions(prefix_len=3, total_len=4), [2])

    def test_prefix_must_have_at_least_one_token_to_predict_from(self) -> None:
        with self.assertRaises(ValueError):
            continuation_logit_positions(prefix_len=0, total_len=4)

    def test_empty_continuation_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            continuation_logit_positions(prefix_len=4, total_len=4)


class ScoreContinuationTests(unittest.TestCase):
    def test_sums_the_log_probability_of_each_target_token(self) -> None:
        log_probs = np.log(np.array([[0.5, 0.25, 0.25], [0.1, 0.1, 0.8]]))

        total, count = score_continuation(log_probs, [2, 1])

        self.assertEqual(count, 2)
        self.assertAlmostEqual(total, math.log(0.25) + math.log(0.1), places=10)

    def test_a_confident_continuation_scores_higher_than_an_unlikely_one(self) -> None:
        log_probs = np.log(np.array([[0.8, 0.1, 0.1], [0.8, 0.1, 0.1]]))

        likely, _ = score_continuation(log_probs, [0, 0])
        unlikely, _ = score_continuation(log_probs, [1, 1])

        self.assertGreater(likely, unlikely)

    def test_row_and_target_count_mismatch_is_rejected(self) -> None:
        log_probs = np.log(np.array([[0.5, 0.25, 0.25]]))

        with self.assertRaises(ValueError):
            score_continuation(log_probs, [2, 1])

    def test_target_outside_the_vocabulary_is_rejected(self) -> None:
        log_probs = np.log(np.array([[0.5, 0.25, 0.25]]))

        with self.assertRaises(ValueError):
            score_continuation(log_probs, [7])


if __name__ == "__main__":
    unittest.main()
