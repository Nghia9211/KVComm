"""No-download contracts for the R1-R10 refactor."""
import importlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import torch
import com_latent as cli
from utils import model_loading, metric_tools
from utils.method_names import latent_method


class RefactorContracts(unittest.TestCase):
    def test_textmas_compatibility_import(self):
        from eval_latent import TextMASEvaluator as old
        from eval_textmas import TextMASEvaluator as new
        self.assertIs(old, new)

    def test_import_does_not_download_nltk(self):
        with patch("nltk.download") as download:
            importlib.reload(metric_tools)
        download.assert_not_called()

    def test_nltk_existing_resources_need_no_download(self):
        with patch("nltk.data.find", return_value=True), patch("nltk.download") as download:
            metric_tools.ensure_nltk_resources()
        download.assert_not_called()

    def test_nltk_missing_resources_are_lazy(self):
        with patch("nltk.data.find", side_effect=LookupError), patch("nltk.download") as download:
            metric_tools.ensure_nltk_resources()
        self.assertEqual(download.call_count, 6)

    def test_model_loading_preserves_options(self):
        with patch.object(model_loading.AutoModelForCausalLM, "from_pretrained") as load:
            model_loading.load_causal_model("model", "cuda:1", revision="pinned")
            load.assert_called_once_with("model", device_map={"": "cuda:1"},
                torch_dtype=torch.bfloat16, attn_implementation="sdpa", revision="pinned")
        self.assertEqual(model_loading.parse_device_map("AUTO"), "auto")

    def test_method_aliases_preserved(self):
        self.assertEqual(latent_method(False), "latent_full")
        self.assertEqual(latent_method(True), "latent_selective")
        self.assertEqual(latent_method(False, response=True), "latentmas_full_kv")
        self.assertEqual(latent_method(True, response=True), "latentmas_selective_kv")

    def test_latent_dispatch_all_selection_paths(self):
        model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=4))
        for options, expected_calls in (
            ({}, 1),
            ({"latent_kv_select": True, "layers_list": [0, 2]}, 1),
            ({"latent_kv_select": True, "random_selection": True, "top_layers": .5}, 1),
            ({"latent_kv_select": True, "top_layers": .5}, 2),
            ({"latent_kv_select": True, "top_layers": .5, "do_layer_curve": True}, 3),
            ({"latent_kv_select": True, "layers_list": [0, 2], "do_layer_curve": True}, 2),
        ):
            with self.subTest(options=options):
                cfg = cli.LatentAlignConfig(**options)
                runner = MagicMock()
                runner.test.return_value = .5
                with patch.object(cli, "LatentMAS"), patch.object(cli, "CVCommunicator"), \
                     patch.object(cli, "LatentCommunicationEvaluator", return_value=runner), \
                     patch.object(cli, "get_top_layers", side_effect=lambda weights, cfg: cfg), \
                     patch.object(cli, "get_layer_ranking", return_value=[0, 2]):
                    cli.run_latent_evaluation(cfg, model, model, object(), object(), None, None, None)
                self.assertEqual(runner.test.call_count, expected_calls)

    def test_main_dispatches_all_baselines(self):
        evaluator = MagicMock()
        evaluator.__len__.return_value = 0
        evaluator.sender_max_tokens = 256
        evaluator.max_tokens = 48
        evaluator.prompt_family = "kvcomm"
        evaluator.prompt_version = "v2"
        evaluator.sender_input_mode = "query_aware_context"
        evaluator.primary_metric = "longbench_qa_f1"
        model = SimpleNamespace(config=SimpleNamespace(_commit_hash=None), eval=lambda: None)
        with tempfile.TemporaryDirectory() as directory:
            cfg = cli.LatentAlignConfig(snapshot_path=directory, do_test_skyline=True,
                do_test_baseline=True, do_test=True, do_test_nld=True)
            with patch.object(cli, "load_tokenizer"), patch.object(cli, "load_causal_model", return_value=model), \
                 patch.object(cli, "get_evaluator", return_value=evaluator), \
                 patch.object(cli, "setup_logging"), patch.object(cli, "log_gpu_info"), \
                 patch.object(cli, "CVCommunicator"), \
                 patch.object(cli, "SkylineEvaluator") as skyline, \
                 patch.object(cli, "BaselineEvaluator") as baseline, \
                 patch.object(cli, "CommunicationEvaluator") as kvcomm, \
                 patch.object(cli, "TextMASEvaluator") as textmas:
                cli.main(cfg)
                for factory in (skyline, baseline, kvcomm, textmas):
                    factory.return_value.test.assert_called_once()


if __name__ == "__main__":
    unittest.main()
