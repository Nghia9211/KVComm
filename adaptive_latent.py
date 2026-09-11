"""Training-free latent stopping. No labels, receiver probes or learned weights.

Features are heuristics, NOT certificates of downstream answer sufficiency.
This module's policy and replay utilities work without torch installed.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from contextlib import contextmanager


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def inference_code_hash(root):
    from pathlib import Path
    root = Path(root)
    paths = [root / name for name in ("models.py", "model_attn.py", "models_latent.py",
        "adaptive_latent.py", "eval.py", "eval_latent.py", "eval_textmas.py", "prompts_latent.py")]
    paths += sorted((root / "dataloader").glob("*.py")) + sorted((root / "utils").glob("*.py"))
    return content_hash({str(path.relative_to(root)).replace("\\", "/"):
                         hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})


def sample_id(item):
    # Hash content, not row index: stable across order, resume and data splits.
    return content_hash({k: v for k, v in item.items() if not k.startswith("_adaptive_")})


def decoding_seed(seed, identity):
    return int(content_hash([int(seed), identity])[:8], 16)


@contextmanager
def sample_rng(seed, identity):
    import torch
    if seed is None:
        yield
        return
    # HF sampling uses torch RNG. Restore it so a probe/sample cannot advance
    # the next sample's stream. CPU and all visible CUDA generators are covered.
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(decoding_seed(seed, identity))
        yield


@dataclass(frozen=True)
class PolicyConfig:
    cosine_distance: float
    log_norm_delta: float | None = None
    value_novelty: float | None = None
    value_window: int = 4
    value_layers: tuple = ()

    def __post_init__(self):
        for name in ("cosine_distance", "log_norm_delta", "value_novelty"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.cosine_distance > 2 or (self.value_novelty is not None and self.value_novelty > 2):
            raise ValueError("Cosine distances must be in [0, 2]")
        if type(self.value_window) is not int or self.value_window < 1:
            raise ValueError("value_window must be a positive integer")
        if any(type(i) is not int or i < 0 for i in self.value_layers):
            raise ValueError("value_layers must contain nonnegative layer indices")
        if len(set(self.value_layers)) != len(self.value_layers):
            raise ValueError("value_layers must be unique")

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as stream:
            document = json.load(stream)
        config = cls(**document["features"])
        return config, document, content_hash(document)


class StopController:
    def __init__(self, policy="fixed", max_steps=5, min_steps=10, interval=5,
                 patience=2, config=None):
        if policy not in ("fixed", "cosine", "hidden_value"):
            raise ValueError(f"Unknown latent policy: {policy}")
        if type(max_steps) is not int or max_steps < 0:
            raise ValueError("latent_steps must be a nonnegative integer")
        if policy != "fixed":
            if not 1 <= min_steps <= max_steps or interval < 1 or patience < 1:
                raise ValueError("Require 1 <= min_steps <= cap, interval >= 1, patience >= 1")
            if config is None:
                raise ValueError("Adaptive stopping requires an explicit policy config")
            if policy == "hidden_value" and (config.value_novelty is None or not config.value_layers):
                raise ValueError("hidden_value requires value_novelty and explicit value_layers")
        self.policy, self.max_steps, self.min_steps = policy, max_steps, min_steps
        self.interval, self.patience, self.config = interval, patience, config
        self.streak = 0
        self.checkpoints = []

    def should_check(self, step):
        return (self.policy != "fixed" and step < self.max_steps and
                step >= self.min_steps and (step - self.min_steps) % self.interval == 0)

    def observe(self, step, features=None):
        if step >= self.max_steps:
            return "fixed_budget" if self.policy == "fixed" else "max_steps"
        if not self.should_check(step):
            return None
        features = features or {}
        names = ["cosine_distance"]
        if self.config.log_norm_delta is not None:
            names.append("log_norm_delta")
        if self.policy == "hidden_value":
            names.append("value_novelty")
        for name in names:
            value = features.get(name)
            if value is not None and not math.isfinite(value):
                raise FloatingPointError(f"Non-finite latent feature {name} at step {step}")
        passed = features.get("valid_hidden", False) and all(
            features.get(name) is not None and features[name] <= getattr(self.config, name)
            for name in names
        )
        self.streak = self.streak + 1 if passed else 0
        self.checkpoints.append({"step": step, "passed": bool(passed), "streak": self.streak})
        return "criterion_met" if self.streak >= self.patience else None


def replay(trace, controller):
    """Replay only observed features; never invent a checkpoint answer/score."""
    by_step = {row["step"]: row for row in trace}
    for step in range(1, controller.max_steps + 1):
        if controller.should_check(step) and step not in by_step:
            raise ValueError(f"Missing feature checkpoint {step}")
        reason = controller.observe(step, by_step.get(step))
        if reason:
            return step, reason
    return 0, "fixed_budget"


def feature_tensor(previous, current, cache, context_length, config, include_values=False):
    """Four small scalars on hidden device; V reductions stay on each layer GPU.

    Novelty = mean over selected layers/heads of minimum cosine distance to
    preceding latent V in a fixed window. Context/sink V is never included.
    Missing history/zero norms invalidate novelty; nonfinite tensors raise when
    materialized. Hidden norm is measured BEFORE realignment.
    """
    import torch
    import torch.nn.functional as F
    p, c = previous.float(), current.float()
    pn, cn = p.norm(dim=-1), c.norm(dim=-1)
    finite = torch.isfinite(p).all() & torch.isfinite(c).all()
    valid = (pn > 1e-12).all() & (cn > 1e-12).all()
    distance = (1 - F.cosine_similarity(p, c, dim=-1)).clamp(0, 2).mean()
    norm_delta = (cn.clamp_min(1e-12).log() - pn.clamp_min(1e-12).log()).abs().mean()
    novelty = c.new_tensor(-1.0)  # missing, not a successful zero novelty
    if include_values:
        scores = []
        for layer in config.value_layers:
            values = cache.value_cache[layer]
            count = values.shape[-2] - context_length
            if count < config.value_window + 1:
                break
            window = values[..., -config.value_window - 1:, :].float()
            vf = torch.isfinite(window).all()
            vv = (window.norm(dim=-1) > 1e-12).all()
            distances = (1 - F.cosine_similarity(window[..., -1:, :], window[..., :-1, :], dim=-1)).clamp(0, 2)
            score = distances.min(dim=-1).values.mean()
            scores.append(torch.where(vf, torch.where(vv, score, score.new_tensor(-1.0)),
                                      score.new_tensor(float("nan"))).to(c.device))
        if len(scores) == len(config.value_layers) and scores:
            stacked = torch.stack(scores)
            novelty = torch.where(torch.isfinite(stacked).all(),
                torch.where((stacked < 0).any(), c.new_tensor(-1.0), stacked.mean()),
                c.new_tensor(float("nan")))
    valid_flag = torch.where(finite, valid.float(), c.new_tensor(float("nan")))
    return torch.stack([distance, norm_delta, novelty, valid_flag]).detach()


def decode_features(values):
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError("Non-finite latent hidden/value state")
    return {"cosine_distance": values[0], "log_norm_delta": values[1],
            "value_novelty": None if values[2] < 0 else values[2],
            "valid_hidden": bool(values[3])}


def synchronize_models(*models):
    import torch
    devices = {p.device for model in models for p in model.parameters() if p.is_cuda}
    for device in devices:
        torch.cuda.synchronize(device)


def cache_payload(cache, layers):
    """Logical routed bytes vs storage retained by slices, BEFORE B mutates KV.

    Mirrors original Mode 2 (layer zero always full, unselected layers keep a
    one-token sink). Neither quantity is measured network traffic.
    """
    logical, storage = 0, {}
    selected = set(layers) | {0}
    for index, pair in enumerate(zip(cache.key_cache, cache.value_cache)):
        for tensor in pair:
            routed = tensor if index in selected else tensor[..., :1, :]
            logical += routed.numel() * routed.element_size()
            backing = routed.untyped_storage()
            storage[(str(routed.device), backing.data_ptr())] = backing.nbytes()
    return {"logical_payload_bytes": logical, "unique_retained_storage_bytes": sum(storage.values())}


def validate_runtime(cfg):
    """Called before model loading by CLI; also usable in dependency-light tests."""
    config, document, digest = None, None, None
    if cfg.latent_policy_config:
        config, document, digest = PolicyConfig.load(cfg.latent_policy_config)
    StopController(cfg.latent_step_policy, cfg.latent_steps, cfg.min_latent_steps,
                   cfg.latent_check_interval, cfg.latent_patience, config)
    adaptive = cfg.latent_step_policy != "fixed"
    if adaptive:
        if not cfg.do_test_latent:
            raise ValueError("Adaptive v1 supports only latent Mode 1/2")
        if cfg.model_A != cfg.model_B or "qwen3" not in cfg.model_A.lower():
            raise ValueError("Adaptive v1 requires identical Qwen3 A/B models")
        if cfg.do_layer_curve or cfg.random_selection or cfg.top_layers > 0:
            raise ValueError("Freeze Mode 2 layers: no auto ranking, random selection or layer curve")
        if cfg.latent_kv_select and (cfg.layers_list == [-1] or not cfg.shift_back):
            raise ValueError("Adaptive Mode 2 requires explicit layers_list and shift_back")
        expected = document.get("controller")
        actual = dict(policy=cfg.latent_step_policy, max_steps=cfg.latent_steps,
                      min_steps=cfg.min_latent_steps, interval=cfg.latent_check_interval,
                      patience=cfg.latent_patience)
        if expected and expected != actual:
            raise ValueError(f"CLI does not match locked policy controller settings: {expected}")
        provenance = document.get("provenance", {})
        if provenance.get("mode") == "m2" and cfg.latent_kv_select:
            if set(provenance["layers_list"]) != set(cfg.layers_list):
                raise ValueError("Mode 2 layers differ from the calibrated policy's frozen layers")
    if cfg.batch_size != 1 and (adaptive or cfg.per_sample_seed or cfg.profile_timing):
        raise ValueError("Adaptive/per-sample RNG/timing require batch_size=1")
    return config, document, digest
