"""Retained entry points and explicit errors for retired experiment flags.

No datasets/model weights are downloaded. Tests also guard against accidental
NLTK downloads. Run from KVComm with the normal test environment.
"""
import contextlib
import importlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from adaptive_latent import inference_code_hash, validate_runtime

ROOT = Path(__file__).resolve().parents[1]


class RuntimeCleanupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch("nltk.download", return_value=True):
            cls.cli = importlib.import_module("com_latent")

    def parse(self, *args):
        with patch("sys.argv", ["com_latent.py", *args]):
            return self.cli.parse_args()

    def test_retained_cli_modes_parse_and_validate(self):
        for args in (("--do_test_latent",),
                     ("--do_test_latent", "--latent_kv_select", "--top_layers", "0.7"),
                     ("--do_test_latent", "--latent_kv_select", "--layers_list", "0", "3"),
                     ("--do_test",), ("--do_test_nld",),
                     ("--do_test_skyline",), ("--do_test_baseline",)):
            with self.subTest(args=args):
                cfg = self.parse(*args)
                validate_runtime(cfg)
                self.assertFalse(hasattr(cfg, "dual_kv_select"))
                self.assertFalse(hasattr(cfg, "track_convergence"))

    def test_adaptive_full_and_selected_cli_remain_supported(self):
        base = ("--model_A", "Qwen/Qwen3-4B", "--model_B", "Qwen/Qwen3-4B",
                "--do_test_latent", "--shift_back", "--latent_steps", "8",
                "--latent_step_policy", "cosine", "--min_latent_steps", "2",
                "--latent_check_interval", "1", "--latent_patience", "2",
                "--latent_policy_config", str(ROOT / "tests/fixtures/adaptive_smoke_policy.json"))
        for selection in ((), ("--latent_kv_select", "--layers_list", "0", "3")):
            cfg = self.parse(*base, *selection)
            config, document, digest = validate_runtime(cfg)
            self.assertEqual(config.cosine_distance, 2.)

    def test_retired_flags_fail_with_clear_error_before_model_loading(self):
        for flag in ("--dual_kv_select", "--segmented_kv_select", "--split_ratio",
                     "--context_top_ratio", "--latent_top_ratio", "--track_convergence"):
            with self.subTest(flag=flag), patch.object(self.cli.AutoModelForCausalLM, "from_pretrained") as load:
                error = io.StringIO()
                with contextlib.redirect_stderr(error), self.assertRaises(SystemExit) as caught:
                    self.parse(flag)
                self.assertEqual(caught.exception.code, 2)
                self.assertIn("has been removed", error.getvalue())
                load.assert_not_called()

    def test_source_hash_no_longer_requires_removed_module(self):
        self.assertFalse((ROOT / "segmented_kv.py").exists())
        self.assertEqual(len(inference_code_hash(ROOT)), 64)

    def test_other_entry_points_and_legacy_baselines_still_import(self):
        with patch("nltk.download", return_value=True):
            for name in ("com", "com_ms", "com_online", "models_ac", "models_cipher"):
                with self.subTest(name=name):
                    importlib.import_module(name)

    def test_sweep_active_modes_and_python_resolution(self):
        source = (ROOT / "sweep_latent.sh").read_text(encoding="utf-8")
        self.assertIn("all) MODES=(m3 textmas m1 m2)", source)
        self.assertNotIn("/mnt/disk2/miniconda3", source)
        self.assertNotIn("args+=(--track_convergence)", source)
        self.assertNotIn("--do_test_latent --dual_kv_select", source)
        self.assertNotIn("--do_test_latent --segmented_kv_select", source)


if __name__ == "__main__":
    unittest.main()
