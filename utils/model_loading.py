"""Shared loading defaults; callers retain ownership of initialization order."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_device_map(device):
    return "auto" if device.lower() == "auto" else {"": device}


def load_tokenizer(name, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_model(name, device, **kwargs):
    return AutoModelForCausalLM.from_pretrained(
        name, device_map=parse_device_map(device),
        torch_dtype=torch.bfloat16, attn_implementation="sdpa", **kwargs)
