"""Dependency-light policy/data/replay tests: python -m unittest discover ..."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from adaptive_latent import PolicyConfig, StopController, content_hash, decoding_seed, replay, sample_id, validate_runtime

ROOT = Path(__file__).resolve().parents[1]


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


samples = module_at("latent_samples", "utils/latent_samples.py")
analysis = module_at("adaptive_analysis", "scripts/analyze_adaptive_latent.py")
profiler = module_at("adaptive_profile", "scripts/profile_adaptive_latent.py")
response_logging = module_at("adaptive_logging", "utils/response_logging.py")


def features(distance=0.0, **kwargs):
    return dict(cosine_distance=distance, valid_hidden=True, **kwargs)


class PolicyTests(unittest.TestCase):
    def controller(self, **kwargs):
        args = dict(policy="cosine", max_steps=23, min_steps=10, interval=5,
                    patience=2, config=PolicyConfig(.01))
        args.update(kwargs)
        return StopController(**args)

    def test_fixed_zero(self):
        self.assertEqual(StopController(max_steps=0).observe(0), "fixed_budget")

    def test_earliest_stop_counts_completed_forwards(self):
        c = self.controller()
        for n in range(1, 15):
            self.assertIsNone(c.observe(n, features()))
        self.assertEqual(c.observe(15, features()), "criterion_met")
        self.assertEqual([r["step"] for r in c.checkpoints], [10, 15])

    def test_failed_check_resets_patience(self):
        c = self.controller(max_steps=30)
        for n, distance in ((10, 0), (15, .1), (20, 0)):
            self.assertIsNone(c.observe(n, features(distance)))
        self.assertEqual(c.observe(25, features()), "criterion_met")

    def test_cap_has_priority(self):
        c = self.controller(max_steps=15)
        c.observe(10, features())
        self.assertEqual(c.observe(15, features()), "max_steps")

    def test_unaligned_cap(self):
        c = self.controller()
        self.assertEqual(c.observe(23, features(.8)), "max_steps")

    def test_min_equals_max(self):
        c = self.controller(min_steps=23)
        self.assertFalse(any(c.should_check(n) for n in range(1, 24)))
        self.assertEqual(c.observe(23), "max_steps")

    def test_no_state_shared_between_samples(self):
        c = self.controller()
        c.observe(10, features())
        fresh = self.controller()
        self.assertEqual(fresh.streak, 0)
        self.assertEqual(fresh.checkpoints, [])

    def test_invalid_configurations(self):
        for kwargs in (dict(max_steps=-1), dict(min_steps=0), dict(min_steps=24),
                       dict(interval=0), dict(patience=0), dict(config=None),
                       dict(policy="learned"), dict(policy="hidden_value")):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.controller(**kwargs)

    def test_invalid_feature_configuration(self):
        for kwargs in (dict(cosine_distance=float("nan")), dict(cosine_distance=3),
                       dict(cosine_distance=-1), dict(value_window=0), dict(value_layers=[0, 0])):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PolicyConfig(**dict({"cosine_distance": .01}, **kwargs))

    def test_nan_and_inf_are_errors(self):
        for value in (float("nan"), float("inf")):
            with self.assertRaises(FloatingPointError):
                self.controller().observe(10, features(value))

    def test_zero_hidden_norm_cannot_converge(self):
        c = self.controller(patience=1)
        row = features()
        row["valid_hidden"] = False
        self.assertIsNone(c.observe(10, row))

    def test_value_history_is_required(self):
        config = PolicyConfig(.01, value_novelty=.01, value_layers=(0,))
        c = self.controller(policy="hidden_value", config=config, patience=1)
        self.assertIsNone(c.observe(10, features(value_novelty=None)))
        self.assertEqual(c.observe(15, features(value_novelty=0)), "criterion_met")

    def test_norm_is_optional_and_independent(self):
        c = self.controller(config=PolicyConfig(.01, log_norm_delta=.1), patience=1)
        self.assertIsNone(c.observe(10, features(log_norm_delta=.2)))
        self.assertEqual(c.observe(15, features(log_norm_delta=.05)), "criterion_met")

    def test_replay_requires_every_decision_feature(self):
        with self.assertRaisesRegex(ValueError, "Missing feature"):
            replay([], self.controller())
        trace = [dict(step=n, **features()) for n in range(1, 24)]
        self.assertEqual(replay(trace, self.controller()), (15, "criterion_met"))

    def test_runtime_rejects_unsupported_combinations_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            path.write_text(json.dumps({"features": {"cosine_distance": .01}}), encoding="utf-8")
            values = dict(latent_policy_config=str(path), latent_step_policy="cosine", latent_steps=80,
                          min_latent_steps=10, latent_check_interval=5, latent_patience=2,
                          do_test_latent=True, dual_kv_select=False, segmented_kv_select=False,
                          model_A="Qwen/Qwen3-4B", model_B="Qwen/Qwen3-4B", do_layer_curve=False,
                          random_selection=False, top_layers=0, latent_kv_select=True, layers_list=[1, 2],
                          shift_back=True, batch_size=1, per_sample_seed=False, profile_timing=False)
            validate_runtime(SimpleNamespace(**values))
            for override in (dict(batch_size=2), dict(top_layers=.7), dict(dual_kv_select=True),
                             dict(segmented_kv_select=True), dict(shift_back=False), dict(layers_list=[-1]),
                             dict(model_B="other"), dict(latent_policy_config="")):
                with self.subTest(override=override), self.assertRaises(ValueError):
                    validate_runtime(SimpleNamespace(**dict(values, **override)))


class ManifestAndAnalysisTests(unittest.TestCase):
    def test_ids_and_seeds_stable_across_key_order(self):
        self.assertEqual(sample_id({"a": 1, "b": 2}), sample_id({"b": 2, "a": 1}))
        self.assertNotEqual(decoding_seed(42, "x"), decoding_seed(42, "y"))

    def test_manifest_roundtrip_excludes_historical(self):
        doc = samples.make_manifest([{"id": str(i)} for i in range(10)], "test", 2, 2, 2, historical_count=3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.json"
            path.write_text(json.dumps(doc), encoding="utf-8")
            items, loaded = samples.load_manifest(path, "test", "holdout")
            self.assertEqual(len(items), 2)
            self.assertTrue(all(int(item["id"]) >= 3 for item in items))
            with self.assertRaises(ValueError):
                samples.load_manifest(path, "other_task", "holdout")
            doc["splits"]["validation"] = doc["splits"]["calibration"]
            doc["manifest_hash"] = content_hash({k: v for k, v in doc.items() if k != "manifest_hash"})
            path.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlapping"):
                samples.load_manifest(path, "test", "holdout")

    def test_resume_refuses_different_hash_and_duplicate_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.jsonl"
            row = dict(config_hash="a", sample_id="one", mode="m1", steps=5, status="error")
            profiler.append_row(path, row)
            self.assertEqual(profiler.read_completed(path, "a"), set())
            row["status"] = "ok"
            profiler.append_row(path, row)
            self.assertEqual(profiler.read_completed(path, "a"), {("one", "m1", 5)})
            with self.assertRaisesRegex(ValueError, "mismatch"):
                profiler.read_completed(path, "b")
            profiler.append_row(path, row)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                profiler.read_completed(path, "a")

    def test_replay_never_invents_unobserved_B_answer(self):
        matrix = {"one": {20: {"adaptive": {"feature_trace": [dict(step=n, **features()) for n in range(1, 21)]}}}}
        spec = dict(policy="cosine", max_steps=20, min_steps=10, interval=5, patience=2)
        with self.assertRaisesRegex(ValueError, "no B answer"):
            analysis.evaluate_policy(matrix, spec, PolicyConfig(.01))

    def test_bootstrap_is_paired(self):
        rows = [{"sample_id": str(i), "score": .8} for i in range(5)]
        refs = [dict(row, score=.5) for row in rows]
        result = analysis.bootstrap_delta(rows, refs, repeats=20)
        self.assertAlmostEqual(result["delta_score"], .3)
        with self.assertRaises(ValueError):
            analysis.bootstrap_delta(rows, list(reversed(refs)), repeats=20)

    def test_response_schema_preserves_actual_steps_per_sample(self):
        evaluator = SimpleNamespace(name="test", prompt_family="kvcomm", prompt_version="v2", sender_input_mode="query_aware_context")
        for n in (10, 35):
            row = response_logging.build_response_record(idx=0, evaluator=evaluator, item={}, method="latent",
                response="answer", model_a_prompt="A", model_b_prompt="B", generated_tokens_a=n,
                generated_tokens_b=9, latent_steps=n,
                adaptive_stats=dict(sample_id=str(n), actual_steps=n, configured_max_steps=80))
            row = json.loads(json.dumps(row))
            self.assertEqual(row["latent"]["steps"], n)
            self.assertEqual(row["adaptive"]["actual_steps"], n)
            self.assertEqual(row["token_counts"]["generated_tokens_B"], 9)
            self.assertEqual(row["schema_version"], "v2")

    def test_analyzer_complete_calibration_to_locked_json(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for split in ("calibration", "validation"):
                identities = [split + str(i) for i in range(3)]
                profile = dict(task="fixture", manifest_hash="data", split=split,
                    sample_ids=identities, steps=[0, 1, 2, 3, 4], modes=["m1"], layers_list=None,
                    features={"cosine_distance": 0., "value_layers": [], "value_window": 4},
                    model="Qwen/Qwen3-fixture", source_hash="code", do_sample=False, seed=42)
                path = directory / (split + ".jsonl")
                for identity in identities:
                    for n in profile["steps"]:
                        profiler.append_row(path, dict(config_hash=content_hash(profile), profile=profile,
                            sample_id=identity, mode="m1", steps=n, split=split, status="ok",
                            score=.9 if n == 2 else .8,
                            adaptive=dict(end_to_end_ms=10 + 10 * n, generated_B_token_count_raw=3,
                                          feature_trace=[dict(step=s, **features(.5 if s < 2 else 0.))
                                                         for s in range(1, n + 1)])))
            policy = directory / "policy.json"
            report = directory / "report.json"
            argv = ["analyze", "--calibration", str(directory / "calibration.jsonl"),
                    "--validation", str(directory / "validation.jsonl"), "--output_policy", str(policy),
                    "--report", str(report), "--max_steps", "4", "--min_steps", "1", "--interval", "1",
                    "--patience", "1", "--cosine_thresholds", ".01", "--budget_fixed_steps", "4"]
            with patch("sys.argv", argv), patch("builtins.print"):
                analysis.main()
            config, document, digest = PolicyConfig.load(policy)
            self.assertEqual(document["controller"]["max_steps"], 4)
            result = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(result["validation"]["mean_steps"], 2)
            self.assertEqual(document["status"], "offline_calibrated_candidate_not_online_validated")
            self.assertIn("validation_sample_ids", document["provenance"])

    def test_analysis_rejects_incomplete_profile_even_if_missing_whole_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.jsonl"
            profile = dict(sample_ids=["one", "two"], modes=["m1"], steps=[0])
            profiler.append_row(path, dict(config_hash=content_hash(profile), profile=profile,
                status="ok", sample_id="one", mode="m1", steps=0, score=1.,
                adaptive={"end_to_end_ms": 1.}))
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                analysis.load_profiles(path)


if __name__ == "__main__":
    unittest.main()
