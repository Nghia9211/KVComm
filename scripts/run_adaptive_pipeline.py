"""End-to-end policy calibration and held-out online evaluation (no training).

Fresh output directory per invocation; existing files are never overwritten.
Use the reference profiler directly to resume interrupted profiling cells.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['policy'], default='policy')
    p.add_argument('--tasks', nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--kv_mode', choices=['m1', 'm2'], default='m1')
    p.add_argument('--policy', choices=['cosine', 'hidden_value'], default='cosine')
    p.add_argument('--model', default='Qwen/Qwen3-4B')
    p.add_argument('--device', default='auto')
    p.add_argument('--device_B', default='auto')
    p.add_argument('--layers_list', type=int, nargs='+')
    p.add_argument('--value_layers', type=int, nargs='+', default=[0, 17, 35])
    p.add_argument('--value_window', type=int, default=4)
    for name, default in [('calibration', 5), ('validation', 5), ('holdout', 10),
                          ('max_steps', 80), ('min_steps', 10), ('interval', 5),
                          ('patience', 2), ('budget_fixed_steps', 40), ('seed', 42),
                          ('warmup', 1), ('max_tokens_B', 0), ('max_input_length', 64000)]:
        p.add_argument('--' + name, type=int, default=default)
    p.add_argument('--historical_count', type=int)
    p.add_argument('--fixed_steps', type=int, nargs='+', default=[0, 10, 20, 40, 80])
    p.add_argument('--b_think', choices=['auto', 'yes', 'no'], default='auto')
    p.add_argument('--metric', help='item_metrics key; otherwise inferred from task primary metric')
    p.add_argument('--time_tolerance', type=float, default=0.0)
    p.add_argument('--dry_run', action='store_true')
    a = p.parse_args(argv)
    if min(a.calibration, a.validation, a.holdout, a.interval, a.patience) <= 0:
        p.error('Split sizes, interval and patience must be positive')
    if not 1 <= a.min_steps <= a.max_steps or min(a.fixed_steps + [a.budget_fixed_steps, a.warmup]) < 0:
        p.error('Invalid step limits or warmup')
    if a.time_tolerance < 0 or a.value_window < 1 or (a.historical_count is not None and a.historical_count < 0):
        p.error('Invalid tolerance/window/historical count')
    if a.kv_mode == 'm2' and (not a.layers_list or min(a.layers_list) < 0):
        p.error('m2 requires frozen explicit --layers_list; no automatic layer ranking')
    if len(set(a.tasks)) != len(a.tasks) or any(not t.replace('_', '').isalnum() for t in a.tasks):
        p.error('Tasks must be unique simple task names')
    if 'qwen3' not in a.model.lower():
        p.error('Adaptive pipeline requires Qwen3')
    return a


def command(script, *args):
    return [sys.executable, str(ROOT / script), *map(str, args)]


def run(cmd, dry=False):
    print(shlex.join(cmd), flush=True)
    if not dry:
        subprocess.run(cmd, cwd=ROOT, check=True)


def read_run(directory):
    paths = list(directory.glob('*/latent_responses.jsonl'))
    if len(paths) != 1:
        raise ValueError(f'Expected exactly one completed evaluation under {directory}')
    return paths[0]


def metric_key(manifest, response_path, override):
    row = json.loads(response_path.read_text(encoding='utf-8').splitlines()[0])
    metrics = row['item_metrics']
    primary = manifest['primary_metric']
    candidate = override or {'longbench_qa_f1': 'longbench_f1'}.get(primary, primary)
    if candidate not in metrics:
        raise ValueError(f'Cannot infer metric {candidate!r}; use --metric from {list(metrics)}')
    return candidate


def task_pipeline(a, task):
    directory = a.output.resolve() / task
    if not a.dry_run:
        directory.mkdir()
    manifest = directory / 'samples.json'
    policy = directory / 'policy.json'
    # Include every possible checkpoint, cap and all fixed references.
    grid = sorted(set([0, a.max_steps, a.budget_fixed_steps, *a.fixed_steps,
                       *range(a.min_steps, a.max_steps + 1, a.interval)]))
    think = a.b_think == 'yes' or (a.b_think == 'auto' and task in {
        'medqa', 'tmath', 'aime2024', 'aime2025', 'gsm8k', 'mbppplus',
        'humanevalplus', 'arc_easy', 'arc_challenge', 'gpqa'})
    prepare = command('scripts/profile_adaptive_latent.py', '--task', task,
        '--prepare_manifest', manifest, '--calibration', a.calibration,
        '--validation', a.validation, '--holdout', a.holdout, '--seed', a.seed)
    if a.historical_count is not None:
        prepare += ['--historical_count', str(a.historical_count)]
    run(prepare, a.dry_run)
    for split in ['calibration', 'validation']:
        cmd = command('scripts/profile_adaptive_latent.py', '--task', task,
            '--sample_manifest', manifest, '--split', split, '--output', directory / (split + '.jsonl'),
            '--model', a.model, '--device', a.device, '--device_B', a.device_B,
            '--seed', a.seed, '--warmup', a.warmup, '--max_tokens_B', a.max_tokens_B,
            '--max_input_length', a.max_input_length, '--value_window', a.value_window,
            '--modes', a.kv_mode, '--value_layers', *a.value_layers, '--steps', *grid)
        if a.kv_mode == 'm2':
            cmd += ['--layers_list', *map(str, a.layers_list)]
        if think:
            cmd += ['--allow_b_think']
        run(cmd, a.dry_run)
    run(command('scripts/analyze_adaptive_latent.py',
        '--calibration', directory / 'calibration.jsonl', '--validation', directory / 'validation.jsonl',
        '--mode', a.kv_mode, '--policy', a.policy, '--max_steps', a.max_steps,
        '--min_steps', a.min_steps, '--interval', a.interval, '--patience', a.patience,
        '--budget_fixed_steps', a.budget_fixed_steps, '--time_tolerance', a.time_tolerance,
        '--output_policy', policy, '--report', directory / 'offline_report.json'), a.dry_run)
    if not a.dry_run and not policy.is_file():
        raise ValueError('No feasible policy produced; stopping before holdout (do not tune on holdout)')
    common = command('com_latent.py', '--model_A', a.model, '--model_B', a.model,
        '--device', a.device, '--device_B', a.device_B, '--test_task', task,
        '--do_test_latent', '--batch_size', 1, '--shift_back', '--sample_manifest', manifest,
        '--sample_split', 'holdout', '--greedy', '--per_sample_seed', '--profile_timing',
        '--seed', a.seed, '--latent_warmup', a.warmup, '--max_tokens_B', a.max_tokens_B,
        '--max_input_length', a.max_input_length)
    if think:
        common += ['--allow_b_think']
    if a.kv_mode == 'm2':
        common += ['--latent_kv_select', '--top_layers', '0', '--layers_list', *map(str, a.layers_list)]
    refs = sorted(set(a.fixed_steps + [a.budget_fixed_steps]))
    for label, steps in [('adaptive', a.max_steps), *[(f'fixed_{n}', n) for n in refs]]:
        cmd = common + ['--snapshot_path', str(directory / label), '--latent_steps', str(steps),
                        '--latent_step_policy', a.policy if label == 'adaptive' else 'fixed']
        if label == 'adaptive':
            cmd += ['--latent_policy_config', str(policy), '--min_latent_steps', str(a.min_steps),
                    '--latent_check_interval', str(a.interval), '--latent_patience', str(a.patience)]
        run(cmd, a.dry_run)
    if a.dry_run:
        print(f'[compare] adaptive vs fixed {refs}; metric inferred from output; reports in {directory}')
        return
    adaptive = read_run(directory / 'adaptive')
    am = json.loads((adaptive.parent / 'manifest.json').read_text(encoding='utf-8'))
    profile = json.loads((directory / 'calibration.jsonl').read_text(encoding='utf-8').splitlines()[0])['profile']
    if any(am[key] != profile['resolved_commit'] for key in ['model_A_commit', 'model_B_commit']):
        raise ValueError('Model revision changed between profiling and evaluation; comparison aborted')
    metric = metric_key(am, adaptive, a.metric)
    for n in refs:
        run(command('scripts/compare_adaptive_runs.py', '--fixed', read_run(directory / f'fixed_{n}'),
            '--adaptive', adaptive, '--metric', metric, '--time_tolerance', a.time_tolerance,
            '--output', directory / f'comparison_fixed_{n}.json'))


def main(argv=None):
    a = parse_args(argv)
    if not a.dry_run:
        a.output.mkdir(parents=True, exist_ok=False)
        (a.output / 'pipeline_config.json').write_text(
            json.dumps(vars(a), default=str, indent=2), encoding='utf-8')
    for task in a.tasks:
        task_pipeline(a, task)
    print('Pipeline complete' if not a.dry_run else 'Dry-run complete: no files/models/datasets created')


if __name__ == '__main__':
    main()
