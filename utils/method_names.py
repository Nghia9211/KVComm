"""Stable method names: preserve manifest and response aliases for old readers."""

def latent_method(selective, *, response=False):
    if response:
        return "latentmas_selective_kv" if selective else "latentmas_full_kv"
    return "latent_selective" if selective else "latent_full"
