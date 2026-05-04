"""Equivalence tests for the batched transformers runner.

The patched _build_transformers_runner adds left-padded batching for the
non-streaming code path. At temperature=0 (greedy decoding), the same prompts
must produce identical outputs whether they go through one batch or many
single-prompt forward passes.

The integration tests below build a tiny synthetic causal LM (no network) so
they run on Mac CPU as part of the standard `python -m unittest` suite.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

try:
    import torch  # noqa: F401
    from transformers import (  # noqa: F401
        AutoModelForCausalLM,
        AutoTokenizer,
        GPT2Config,
        GPT2LMHeadModel,
        GPT2Tokenizer,
    )

    _HAVE_TRANSFORMERS = True
except ImportError:
    _HAVE_TRANSFORMERS = False


# `gpt2` ships an offline-friendly tokenizer config; the cached files are tiny.
# Tests that need a tokenizer pull from this name. If the cache miss + network
# fetch is undesirable in a given environment, set THESIS_EVAL_SKIP_NET_TESTS=1.
_GPT2_TOKENIZER_REF = "gpt2"


def _build_tiny_causal_lm_dir(tmp: Path) -> str:
    """Materialize a tiny GPT-2 causal LM with random weights into ``tmp``.

    Returns the path that AutoModelForCausalLM/AutoTokenizer can both consume.
    The model is small (~50 KB on disk after safetensors) and deterministic
    given a fixed seed.
    """
    import torch
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

    tokenizer = AutoTokenizer.from_pretrained(_GPT2_TOKENIZER_REF)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(tmp)

    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=128,
        n_embd=32,
        n_layer=2,
        n_head=4,
    )
    torch.manual_seed(0)
    model = GPT2LMHeadModel(config)
    model.save_pretrained(tmp)
    return str(tmp)


@unittest.skipUnless(_HAVE_TRANSFORMERS, "torch + transformers not installed")
@unittest.skipIf(
    os.environ.get("THESIS_EVAL_SKIP_NET_TESTS") == "1",
    "skipped: THESIS_EVAL_SKIP_NET_TESTS=1 (gpt2 tokenizer fetch disabled)",
)
class TestBatchedTransformersRunner(unittest.TestCase):
    """Verify batched generation is functionally equivalent to sequential."""

    def setUp(self) -> None:
        # Each test uses its own tmp dir so model artifacts don't leak between tests.
        self._tmp = tempfile.TemporaryDirectory()
        self.model_dir = _build_tiny_causal_lm_dir(Path(self._tmp.name))
        self.prompts = [
            "The quick brown fox",
            "Once upon a time, in a far away kingdom,",
            "To be or not to be,",
            "Hello",
        ]

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self, *, batch_size: int) -> list[str]:
        from thesis_eval.models.generation import _build_transformers_runner

        runner = _build_transformers_runner(
            self.model_dir,
            max_tokens=8,
            temperature=0.0,
            device="cpu",
            dtype="float32",
            trust_remote_code=False,
            stream=False,
            batch_size=batch_size,
            max_input_length=None,
        )
        return runner(self.prompts)

    def test_batched_equals_sequential_at_temp_zero(self) -> None:
        sequential = self._run(batch_size=1)
        batched = self._run(batch_size=2)
        self.assertEqual(len(sequential), len(self.prompts))
        self.assertEqual(len(batched), len(self.prompts))
        # Greedy decoding with left-padded attention masks must be deterministic
        # and identical across the two paths. Decoded text comparison is the
        # cheap, robust signal.
        self.assertEqual(sequential, batched)

    def test_batched_full_chunk_equals_sequential(self) -> None:
        # batch_size >= len(prompts): the runner does one forward pass.
        sequential = self._run(batch_size=1)
        single_batch = self._run(batch_size=len(self.prompts))
        self.assertEqual(sequential, single_batch)

    def test_streaming_with_batch_size_gt_one_is_rejected(self) -> None:
        from thesis_eval.models.generation import _build_transformers_runner

        with self.assertRaisesRegex(RuntimeError, "Streaming generation requires batch_size=1"):
            _build_transformers_runner(
                self.model_dir,
                max_tokens=4,
                temperature=0.0,
                device="cpu",
                dtype="float32",
                stream=True,
                batch_size=2,
            )


@unittest.skipUnless(_HAVE_TRANSFORMERS, "torch + transformers not installed")
class TestGpuBatchSizeResolver(unittest.TestCase):
    """Verify the new --gpu-batch-size resolver applies the right defaults."""

    def test_transformers_default_for_generate(self) -> None:
        from thesis_eval.cli import (
            _DEFAULT_TRANSFORMERS_GPU_BATCH_SIZE_GENERATE,
            _resolve_gpu_batch_size,
        )

        self.assertEqual(
            _resolve_gpu_batch_size(None, "transformers", default=_DEFAULT_TRANSFORMERS_GPU_BATCH_SIZE_GENERATE),
            _DEFAULT_TRANSFORMERS_GPU_BATCH_SIZE_GENERATE,
        )

    def test_transformers_default_for_belebele(self) -> None:
        from thesis_eval.cli import (
            _DEFAULT_TRANSFORMERS_GPU_BATCH_SIZE_BELEBELE,
            _resolve_gpu_batch_size,
        )

        self.assertEqual(
            _resolve_gpu_batch_size(None, "transformers", default=_DEFAULT_TRANSFORMERS_GPU_BATCH_SIZE_BELEBELE),
            _DEFAULT_TRANSFORMERS_GPU_BATCH_SIZE_BELEBELE,
        )

    def test_vllm_returns_one(self) -> None:
        # vLLM has its own continuous-batching engine; --gpu-batch-size is moot.
        from thesis_eval.cli import _resolve_gpu_batch_size

        self.assertEqual(_resolve_gpu_batch_size(None, "vllm", default=8), 1)

    def test_explicit_value_wins(self) -> None:
        from thesis_eval.cli import _resolve_gpu_batch_size

        self.assertEqual(_resolve_gpu_batch_size(4, "transformers", default=8), 4)
        self.assertEqual(_resolve_gpu_batch_size(4, "vllm", default=8), 4)

    def test_invalid_value_raises(self) -> None:
        from thesis_eval.cli import _resolve_gpu_batch_size

        with self.assertRaisesRegex(ValueError, "--gpu-batch-size must be at least 1"):
            _resolve_gpu_batch_size(0, "transformers", default=8)


if __name__ == "__main__":
    unittest.main()
