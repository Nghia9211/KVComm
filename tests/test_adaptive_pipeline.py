"""Orchestration tests: no model or dataset downloads."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts import run_adaptive_pipeline as pipeline


class PipelineTests(unittest.TestCase):
    def test_regular_modes_bypass_policy_setup(self):
        for mode in ['textmas', 'm1', 'm2', 'm3', 'all', 'both']:
            with self.subTest(mode=mode), patch.object(pipeline.shutil, 'which', return_value='bash'), \
                 patch.object(pipeline.subprocess, 'run') as run, \
                 patch.object(pipeline, 'parse_args') as parse:
                pipeline.main(['--mode', mode, '--tasks', 'hotpotqa', 'gsm8k', '--dry_run'])
                parse.assert_not_called()
                command = run.call_args.args[0]
                self.assertIn('hotpotqa gsm8k', command)
                self.assertIn(mode, command)

    def test_dry_run_is_read_only_and_covers_all_stages(self):
        with tempfile.TemporaryDirectory() as parent:
            output = Path(parent) / 'new'
            with patch.object(pipeline.subprocess, 'run') as run, contextlib.redirect_stdout(io.StringIO()) as log:
                pipeline.main(['--tasks', 'hotpotqa', 'gsm8k', '--output', str(output), '--dry_run'])
            run.assert_not_called()
            self.assertFalse(output.exists())
            text = log.getvalue()
            for name in ['--prepare_manifest', '--split calibration', '--split validation',
                         'analyze_adaptive_latent.py', '--sample_split holdout', '--latent_step_policy cosine']:
                self.assertIn(name, text)
            self.assertNotIn('--latent_trace', text)

    def test_missing_frozen_layers_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            pipeline.parse_args(['--tasks', 'hotpotqa', '--output', 'unused', '--kv_mode', 'm2'])

    def test_existing_directory_not_overwritten(self):
        with tempfile.TemporaryDirectory() as parent, self.assertRaises(FileExistsError):
            pipeline.main(['--tasks', 'hotpotqa', '--output', parent])

    def test_failure_stops_later_stages(self):
        with tempfile.TemporaryDirectory() as parent:
            with patch.object(pipeline, 'run', side_effect=RuntimeError('GPU failure')) as run:
                with self.assertRaises(RuntimeError):
                    pipeline.main(['--tasks', 'hotpotqa', '--output', str(Path(parent) / 'new')])
            self.assertEqual(run.call_count, 1)

    def test_no_policy_never_evaluates_holdout(self):
        with tempfile.TemporaryDirectory() as parent:
            with patch.object(pipeline, 'run') as run, self.assertRaisesRegex(ValueError, 'No feasible policy'):
                pipeline.main(['--tasks', 'hotpotqa', '--output', str(Path(parent) / 'new')])
            self.assertEqual(run.call_count, 4)

    def test_m2_layer_and_decoding_consistency(self):
        args = pipeline.parse_args(['--tasks', 'gsm8k', '--output', 'unused', '--dry_run',
            '--kv_mode', 'm2', '--layers_list', '0', '3', '--min_steps', '7', '--interval', '6'])
        with patch.object(pipeline, 'run') as run, contextlib.redirect_stdout(io.StringIO()):
            pipeline.task_pipeline(args, 'gsm8k')
        commands = [c.args[0] for c in run.call_args_list]
        profiles = [c for c in commands if '--split' in c]
        self.assertEqual(len(profiles), 2)
        for cmd in profiles:
            self.assertIn('--allow_b_think', cmd)
            self.assertIn('13', cmd)
            self.assertEqual(cmd[cmd.index('--layers_list') + 1:], ['0', '3', '--allow_b_think'])
        for cmd in commands:
            if '--sample_split' in cmd:
                self.assertIn('--greedy', cmd)
                self.assertIn('--latent_kv_select', cmd)
                self.assertIn('--allow_b_think', cmd)


if __name__ == '__main__':
    unittest.main()
