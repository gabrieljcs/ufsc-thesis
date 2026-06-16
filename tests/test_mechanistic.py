from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from thesis_eval.config import load_config
from thesis_eval.mechanistic.refusal import (
    analyze_refusal_directions,
    link_refusal_separability_to_asr,
    prepare_refusal_sample,
)


class MechanisticRefusalTests(unittest.TestCase):
    def test_prepare_refusal_sample_balances_only_clear_open_weight_contrasts(self) -> None:
        cfg = load_config()
        rows = [
            {
                "model": "llama3_1_8b_reference",
                "model_access_mode": "open_weight",
                "attack_language": "eng",
                "aligned_language": "eng",
                "prompt_id": "p1",
                "run_id": "r1",
                "prompt_text": "harmful prompt",
                "generation_completed": True,
                "unsafe_binary": 0,
                "refusal_binary": 1,
                "strongreject_score": 0.0,
            },
            {
                "model": "llama3_1_8b_reference",
                "model_access_mode": "open_weight",
                "attack_language": "eng",
                "aligned_language": "eng",
                "prompt_id": "p2",
                "run_id": "r2",
                "prompt_text": "harmful prompt",
                "generation_completed": True,
                "unsafe_binary": 1,
                "refusal_binary": 0,
                "strongreject_score": 1.0,
            },
            {
                "model": "llama3_1_8b_reference",
                "model_access_mode": "open_weight",
                "attack_language": "eng",
                "aligned_language": "eng",
                "prompt_id": "ambiguous",
                "run_id": "r3",
                "prompt_text": "harmful prompt",
                "generation_completed": True,
                "unsafe_binary": 1,
                "refusal_binary": 1,
                "strongreject_score": 1.0,
            },
            {
                "model": "sabia_3",
                "model_access_mode": "api",
                "attack_language": "por",
                "aligned_language": "por",
                "prompt_id": "api",
                "run_id": "r4",
                "prompt_text": "harmful prompt",
                "generation_completed": True,
                "unsafe_binary": 0,
                "refusal_binary": 1,
                "strongreject_score": 0.0,
            },
        ]

        sample = prepare_refusal_sample(
            rows,
            cfg,
            models=["llama3_1_8b_reference", "sabia_3"],
            languages=["eng", "por"],
            max_per_label_per_cell=4,
            seed=0,
        )

        self.assertEqual([row["prompt_id"] for row in sample], ["p1", "p2"])
        self.assertEqual({row["contrast_label"] for row in sample}, {"refusal", "unsafe"})
        self.assertEqual({row["model"] for row in sample}, {"llama3_1_8b_reference"})

    def test_analyze_refusal_directions_separates_synthetic_refusal_axis(self) -> None:
        metadata = [
            {
                "mechanistic_sample_id": "toy:eng:p1:r1",
                "model": "toy",
                "attack_language": "eng",
                "aligned_language": "eng",
                "prompt_id": "p1",
                "run_id": "r1",
                "contrast_label": "refusal",
                "unsafe_binary": 0,
                "refusal_binary": 1,
                "activation_layer_resolved": 1,
            },
            {
                "mechanistic_sample_id": "toy:eng:p2:r2",
                "model": "toy",
                "attack_language": "eng",
                "aligned_language": "eng",
                "prompt_id": "p2",
                "run_id": "r2",
                "contrast_label": "unsafe",
                "unsafe_binary": 1,
                "refusal_binary": 0,
                "activation_layer_resolved": 1,
            },
            {
                "mechanistic_sample_id": "toy:spa:p3:r3",
                "model": "toy",
                "attack_language": "spa",
                "aligned_language": "eng",
                "prompt_id": "p3",
                "run_id": "r3",
                "contrast_label": "refusal",
                "unsafe_binary": 0,
                "refusal_binary": 1,
                "activation_layer_resolved": 1,
            },
            {
                "mechanistic_sample_id": "toy:spa:p4:r4",
                "model": "toy",
                "attack_language": "spa",
                "aligned_language": "eng",
                "prompt_id": "p4",
                "run_id": "r4",
                "contrast_label": "unsafe",
                "unsafe_binary": 1,
                "refusal_binary": 0,
                "activation_layer_resolved": 1,
            },
        ]
        activations = np.asarray(
            [
                [1.0, 0.0],
                [-1.0, 0.0],
                [2.0, 0.0],
                [-2.0, 0.0],
            ],
            dtype=np.float32,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "toy.activations.npz"
            np.savez_compressed(
                path,
                activations=activations,
                metadata_json=np.asarray([json.dumps(row, sort_keys=True) for row in metadata]),
            )
            result = analyze_refusal_directions([path], min_per_label=1)

        summary = result["summary"]
        self.assertEqual(len(summary), 2)
        self.assertTrue(all(row["projection_gap_refusal_minus_unsafe"] > 0 for row in summary))
        self.assertTrue(all(row["auc_refusal_vs_unsafe"] == 1.0 for row in summary))
        self.assertEqual(len(result["projections"]), 4)
        self.assertEqual(list(result["directions"]["models"]), ["toy"])
        self.assertIsNone(result["selected_layer"])  # 2D bundle => no sweep
        spa_similarity = [row for row in result["language_similarity"] if row["attack_language"] == "spa"][0]
        self.assertAlmostEqual(spa_similarity["cosine_with_reference_language"], 1.0)
        self.assertAlmostEqual(spa_similarity["cosine_with_pooled_direction"], 1.0)
        # Cross-lingual transfer: the eng-estimated direction perfectly separates spa, and vice versa.
        self.assertTrue(result["transfer"])
        self.assertTrue(all(row["transfer_auc"] == 1.0 for row in result["transfer"]))

    def test_layer_sweep_selects_the_separating_layer(self) -> None:
        # Two layers: layer 0 is pure noise (no separation), layer 1 separates refusal>0 / unsafe<0.
        metadata = []
        vectors = []
        for prompt_index in range(8):
            label = "refusal" if prompt_index % 2 == 0 else "unsafe"
            sign = 1.0 if label == "refusal" else -1.0
            metadata.append(
                {
                    "mechanistic_sample_id": f"toy:eng:p{prompt_index}:r{prompt_index}",
                    "model": "toy",
                    "attack_language": "eng",
                    "aligned_language": "eng",
                    "prompt_id": f"p{prompt_index}",
                    "run_id": f"r{prompt_index}",
                    "contrast_label": label,
                    "unsafe_binary": 0 if label == "refusal" else 1,
                    "refusal_binary": 1 if label == "refusal" else 0,
                }
            )
            noise_layer = [0.01 * ((prompt_index % 3) - 1), 0.0]
            signal_layer = [sign * (1.0 + 0.1 * prompt_index), 0.0]
            vectors.append([noise_layer, signal_layer])  # shape (L=2, H=2)
        activations = np.asarray(vectors, dtype=np.float32)  # (n, 2, 2)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "toy.layers.npz"
            np.savez_compressed(
                path,
                activations=activations,
                metadata_json=np.asarray([json.dumps(row, sort_keys=True) for row in metadata]),
                layers=np.asarray([10, 11], dtype=np.int64),
            )
            result = analyze_refusal_directions([path], min_per_label=1, seed=0)

        self.assertEqual(result["selected_layer"], 11)  # the separating layer
        self.assertEqual(len(result["layer_sweep"]), 2)
        selected = [row for row in result["layer_sweep"] if row["selected"]]
        self.assertEqual([row["layer"] for row in selected], [11])

    def test_link_refusal_separability_to_asr_computes_spearman(self) -> None:
        summary = [
            {"model": "m", "attack_language": "eng", "auc_refusal_vs_unsafe": 0.9, "n_refusal": 5, "n_unsafe": 5},
            {"model": "m", "attack_language": "spa", "auc_refusal_vs_unsafe": 0.7, "n_refusal": 5, "n_unsafe": 5},
            {"model": "m", "attack_language": "swa", "auc_refusal_vs_unsafe": 0.5, "n_refusal": 5, "n_unsafe": 5},
        ]
        asr_rows = [
            {"model": "m", "attack_language": "eng", "asr": 0.1, "n": 10},
            {"model": "m", "attack_language": "spa", "asr": 0.4, "n": 10},
            {"model": "m", "attack_language": "swa", "asr": 0.8, "n": 10},
        ]
        joined, stats = link_refusal_separability_to_asr(summary, asr_rows)
        self.assertEqual(len(joined), 3)
        self.assertEqual(stats["n_cells"], 3)
        # Higher separability => lower ASR: perfect negative rank correlation.
        self.assertAlmostEqual(stats["spearman_auc_vs_asr"], -1.0)


if __name__ == "__main__":
    unittest.main()

