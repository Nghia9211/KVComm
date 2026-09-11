"""Tiny random Qwen3, no weights/dataset downloads. CPU by default.

Set ADAPTIVE_TEST_DEVICE=cuda:0 for GPU regression, or
ADAPTIVE_TEST_DEVICE=multi for two-GPU Accelerate dispatch smoke.
"""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.cache_utils import DynamicCache

from adaptive_latent import PolicyConfig, cache_payload, decode_features, feature_tensor, sample_rng
from models_latent import LatentMAS
from models import CVCommunicator


def tiny_model(seed):
    torch.manual_seed(seed)
    config = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=8, max_position_embeddings=256,
                        pad_token_id=0, bos_token_id=1, eos_token_id=2,
                        attention_dropout=0.0)
    config._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(config).eval()
    device = os.environ.get("ADAPTIVE_TEST_DEVICE", "cpu")
    if device == "multi":
        if torch.cuda.device_count() < 2:
            raise unittest.SkipTest("multi needs at least two CUDA devices")
        from accelerate import dispatch_model
        model = dispatch_model(model, device_map={"model.embed_tokens": 0, "model.rotary_emb": 0,
            "model.layers.0": 0, "model.layers.1": 0, "model.layers.2": 1, "model.layers.3": 1,
            "model.norm": 1, "lm_head": 1})
    else:
        model.to(device)
    model.name = "Qwen/Qwen3-tiny-test"
    return model


@torch.no_grad()
def reference_fixed(latent, ids, steps):
    """Pre-adaptive forward loop, independent of the controller integration."""
    ids = ids.to(latent.model.get_input_embeddings().weight.device)
    outputs = latent.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                           use_cache=True, output_hidden_states=True, return_dict=True)
    past, hidden = outputs.past_key_values, outputs.hidden_states[-1][:, -1, :]
    for step in range(steps):
        outputs = latent.model(inputs_embeds=latent._apply_realignment(hidden).unsqueeze(1),
            attention_mask=torch.ones((1, ids.shape[-1] + step + 1), device=ids.device, dtype=torch.long),
            past_key_values=past, use_cache=True, output_hidden_states=True, return_dict=True)
        past = latent._normalize_cache_devices(outputs.past_key_values)
        hidden = outputs.hidden_states[-1][:, -1, :]
    return past


class TinyQwenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.a, cls.b = tiny_model(123), tiny_model(456)
        cls.ids = torch.tensor([[1, 8, 12, 6, 9]])

    def assert_cache_equal(self, a, b):
        for pairs in ((a.key_cache, b.key_cache), (a.value_cache, b.value_cache)):
            for x, y in zip(*pairs):
                torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-5)

    def test_fixed_matches_legacy_reference_and_trace_is_observational(self):
        for steps in (0, 1, 4):
            with self.subTest(steps=steps):
                latent = LatentMAS(self.a, latent_steps=steps)
                fixed = latent.run(self.ids)
                reference = reference_fixed(latent, self.ids, steps)
                self.assert_cache_equal(fixed, reference)
                latent.latent_trace = True
                traced = latent.run(self.ids)
                self.assert_cache_equal(fixed, traced)
                self.assertEqual(len(latent.last_run_stats["feature_trace"]), steps)
                self.assertEqual(traced._kvcomm_latent_length, steps)

    def test_forced_stop_no_extra_forwards_and_reset(self):
        latent = LatentMAS(self.a, latent_steps=8, latent_step_policy="cosine",
                          min_latent_steps=2, latent_check_interval=1, latent_patience=2,
                          policy_config=PolicyConfig(2.0))
        calls = []
        hook = self.a.register_forward_hook(lambda *args: calls.append(1))
        try:
            for _ in range(2):
                calls.clear()
                cache = latent.run(self.ids)
                self.assertEqual(len(calls), 4)  # 1 prefill + 3 completed latent steps
                self.assertEqual(cache.get_seq_length(), 8)
                self.assertEqual(cache._kvcomm_latent_length, 3)
                self.assertEqual(latent.last_run_stats["stop_reason"], "criterion_met")
        finally:
            hook.remove()

    def test_full_and_selected_receiver_logits_masks_outputs_match_fixed_n(self):
        for selected in ([0, 1, 2, 3], [1, 3]):  # layer 0 remains full even if omitted
            with self.subTest(selected=selected):
                fixed = LatentMAS(self.a, latent_steps=3).run(self.ids)
                adaptive = LatentMAS(self.a, latent_steps=8, latent_step_policy="cosine",
                    min_latent_steps=2, latent_check_interval=1, latent_patience=2,
                    policy_config=PolicyConfig(2.0)).run(self.ids)
                self.assert_cache_equal(fixed, adaptive)
                cv = CVCommunicator(self.a, self.b, 0, -1, layers_list=selected, shift_back=True)
                routed = cv.prepare_key_cache(adaptive)
                for layer in range(4):
                    self.assertEqual(routed.key_cache[layer].shape[-2], 8 if layer in selected or layer == 0 else 1)
                b_ids = torch.tensor([[1, 5, 7]], device=self.b.get_input_embeddings().weight.device)
                masks = []
                handles = []
                def capture(module, args, kwargs):
                    masks.append((kwargs.get("attention_mask").detach().cpu().clone()
                                  if kwargs.get("attention_mask") is not None else None,
                                  tuple(t.detach().cpu().clone() for t in kwargs.get("position_embeddings", ()))))
                for layer in self.b.model.layers:
                    handles.append(layer.self_attn.register_forward_pre_hook(capture, with_kwargs=True))
                results, geometries = [], []
                try:
                    with torch.no_grad():
                        for cache in (fixed, adaptive):
                            masks.clear()
                            result = cv.generate(b_ids, attention_mask=torch.ones((1, 11), dtype=torch.long, device=b_ids.device),
                                out_A_past_key_values=cache, max_new_tokens=3, do_sample=False,
                                return_dict_in_generate=True, output_scores=True)
                            results.append(result)
                            geometries.append(list(masks))
                finally:
                    for handle in handles:
                        handle.remove()
                self.assertTrue(torch.equal(results[0].sequences, results[1].sequences))
                for x, y in zip(results[0].scores, results[1].scores):
                    torch.testing.assert_close(x, y)
                self.assertEqual(len(geometries[0]), len(geometries[1]))
                for (mask0, pos0), (mask1, pos1) in zip(*geometries):
                    self.assertEqual(mask0 is None, mask1 is None)
                    if mask0 is not None:
                        torch.testing.assert_close(mask0, mask1)
                    for x, y in zip(pos0, pos1):
                        torch.testing.assert_close(x, y)
                # Each independent reference cell owns its A rollout. Handoff
                # to B must not change the sender's cache length or tensors.
                self.assertEqual(fixed.get_seq_length(), 8)
                self.assertEqual(adaptive.get_seq_length(), 8)
                self.assert_cache_equal(fixed, adaptive)

    def test_min_equals_cap_and_adaptive_batch_rejection(self):
        latent = LatentMAS(self.a, latent_steps=3, latent_step_policy="cosine",
                          min_latent_steps=3, policy_config=PolicyConfig(2.0))
        forced = latent.run(self.ids)
        self.assert_cache_equal(forced, LatentMAS(self.a, latent_steps=3).run(self.ids))
        self.assertEqual(latent.last_run_stats["stop_reason"], "max_steps")
        with self.assertRaisesRegex(ValueError, "batch_size=1"):
            latent.run(self.ids.repeat(2, 1))

    def test_hidden_value_waits_for_history_and_matches_fixed_prefix(self):
        latent = LatentMAS(self.a, latent_steps=8, latent_step_policy="hidden_value",
            min_latent_steps=1, latent_check_interval=1, latent_patience=1,
            policy_config=PolicyConfig(2., value_novelty=2., value_window=2, value_layers=(0, 3)),
            latent_trace=True)
        cache = latent.run(self.ids)
        self.assertEqual(cache._kvcomm_latent_length, 3)
        self.assertIsNone(latent.last_run_stats["feature_trace"][0]["value_novelty"])
        self.assertIsNotNone(latent.last_run_stats["feature_trace"][2]["value_novelty"])
        self.assert_cache_equal(cache, LatentMAS(self.a, latent_steps=3).run(self.ids))

    def test_rng_restored_and_seed_is_per_sample(self):
        state = torch.get_rng_state().clone()
        with sample_rng(42, "a"):
            first = torch.rand(4)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        with sample_rng(42, "b"):
            torch.rand(99)
        with sample_rng(42, "a"):
            self.assertTrue(torch.equal(first, torch.rand(4)))

    def test_evaluator_writes_real_lengths_and_raw_B_tokens(self):
        # This test uses no NLTK metrics; guard against accidental downloads.
        with patch("nltk.download", return_value=True):
            from eval_latent import LatentCommunicationEvaluator
        class Dataset:
            name = "fixture"
            prompt_family = "kvcomm"
            prompt_version = "v2"
            sender_input_mode = "query_aware_context"
            primary_metric = "longbench_qa_f1"
            max_tokens = 3
            def __iter__(self):
                return iter([{"id": "one", "answer": "answer"}, {"id": "two", "answer": "answer"}])
            def evaluate_item(self, item, response):
                return {"longbench_f1": 1.0}
            def get_result(self):
                return 1.0
            def get_results(self):
                return {"primary": 1.0}
        class Tokenizer:
            eos_token_id = 2
            def decode(self, *args, **kwargs):
                return "<think>some reasoning</think>answer"
            def encode(self, *args, **kwargs):
                return [9]  # cannot be used as the count of generated IDs
        latent = LatentMAS(self.a, latent_steps=8, latent_step_policy="cosine",
            min_latent_steps=1, latent_check_interval=1, latent_patience=1,
            policy_config=PolicyConfig(.01), greedy=True, sample_seed=42, profile_timing=True)
        cv = CVCommunicator(self.a, self.b, 0, -1, layers_list=[1, 3], shift_back=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "responses.jsonl"
            runner = LatentCommunicationEvaluator(Dataset(), Tokenizer(), False, 200, latent, cv,
                allow_b_think=True, response_log_path=str(path))
            runner.generate_args["eos_token_id"] = None
            from unittest.mock import Mock
            runner.prepare_input_ids = Mock(side_effect=lambda *args: (self.ids, self.ids[:, :3]))
            packed = [torch.tensor([d, 0., -1., 1.]) for d in (.2, 0., .2, .2, 0.)]
            with patch("models_latent.feature_tensor", side_effect=packed):
                runner.test(self.a, cv, batch_size=1)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(runner.prepare_input_ids.call_count, 2)
            self.assertNotIn("segmented_stats", rows[0]["latent"])
            self.assertEqual([row["latent"]["steps"] for row in rows], [2, 3])
            self.assertEqual([row["adaptive"]["actual_cache_length_before_B"] for row in rows], [7, 8])
            self.assertTrue(all(row["token_counts"]["generated_tokens_B"] == 3 for row in rows))
            self.assertTrue(all(row["response"] == "answer" for row in rows))
            self.assertTrue(all(row["adaptive"]["end_to_end_ms"] > 0 for row in rows))


class FeatureTests(unittest.TestCase):
    def cache(self, values):
        cache = DynamicCache()
        tensor = torch.tensor(values, dtype=torch.float32).reshape(1, 1, -1, 2)
        cache.update(tensor.clone(), tensor, 0)
        return cache

    def test_value_excludes_context_and_needs_full_window(self):
        config = PolicyConfig(.1, value_window=2, value_layers=(0,))
        h = torch.tensor([[1., 0.]])
        # Context matches new V, but previous latent window is orthogonal.
        cache = self.cache([[1, 0], [0, 1], [0, 1], [1, 0]])
        row = decode_features(feature_tensor(h, h, cache, 1, config, True).tolist())
        self.assertAlmostEqual(row["value_novelty"], 1)
        row = decode_features(feature_tensor(h, h, cache, 2, config, True).tolist())
        self.assertIsNone(row["value_novelty"])

    def test_raw_hidden_norm_and_invalid_states(self):
        config = PolicyConfig(.1)
        h = torch.tensor([[1., 0.]])
        cache = self.cache([[1, 0]])
        row = decode_features(feature_tensor(h, h * 2, cache, 1, config).tolist())
        self.assertAlmostEqual(row["cosine_distance"], 0)
        self.assertGreater(row["log_norm_delta"], .6)
        row = decode_features(feature_tensor(h, h * 0, cache, 1, config).tolist())
        self.assertFalse(row["valid_hidden"])
        with self.assertRaises(FloatingPointError):
            decode_features(feature_tensor(h, h * float("nan"), cache, 1, config).tolist())

    def test_logical_bytes_not_equal_retained_storage(self):
        cache = self.cache([[1, 0], [0, 1], [1, 1]])
        cache.update(cache.key_cache[0].clone(), cache.value_cache[0].clone(), 1)
        stats = cache_payload(cache, [0])
        self.assertEqual(stats["logical_payload_bytes"], (3 + 1) * 2 * 4 * 2)
        self.assertGreater(stats["unique_retained_storage_bytes"], stats["logical_payload_bytes"])


if __name__ == "__main__":
    unittest.main()
