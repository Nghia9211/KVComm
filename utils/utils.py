import os
import logging
import torch

def log_gpu_info():
    if torch.cuda.is_available():
        logging.info(f"Number of GPU: {torch.cuda.device_count()}")
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'None')}")
    else:
        logging.warning("CUDA is not available!")

def setup_logging(log_file_path: str, log_level: str = "INFO"):
    log_dir = os.path.dirname(log_file_path)
    os.makedirs(log_dir, exist_ok=True)
    
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR
    }
    level = log_level_map.get(log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file_path, encoding='utf-8')
        ]
    )
    
    logging.info(f"Logging setup completed. Log file: {log_file_path}")
    logging.info(f"Log level: {log_level}")


def get_model_short_name(model_path: str) -> str:
    name = model_path.split("/")[-1]
    return name.replace("-", "").replace("_", "").lower()


def get_method_name(cfg) -> str:
    """
    Detect the evaluation method from cfg flags and return a short method label.

    Priority order (first match wins):
      do_test_latent   → LatentMAS   (com_latent.py)
      do_test_cipher   → CIPHER
      do_test_nld      → NLD
      do_test_ac       → AC
      do_test_skyline  → Skyline
      do_test_baseline → Baseline
      do_test          → KVComm
      (fallback)       → KVComm

    Example snapshot folder names produced:
      llama3.23binstruct-to-llama3.23binstruct_from0to26_KVComm_0807_1800
      llama3.23binstruct-to-llama3.23binstruct_from0to26_NLD_0807_1800
      llama3.23binstruct-to-llama3.23binstruct_from0to26_CIPHER_0807_1800
      llama3.23binstruct-to-llama3.23binstruct_from0to26_LatentMAS_lat3_realign_0807_1800
    """
    if getattr(cfg, "do_test_latent", False):
        return "LatentMAS"
    if getattr(cfg, "do_test_cipher", False):
        return "CIPHER"
    if getattr(cfg, "do_test_nld", False):
        return "NLD"
    if getattr(cfg, "do_test_ac", False):
        return "AC"
    if getattr(cfg, "do_test_skyline", False):
        return "Skyline"
    if getattr(cfg, "do_test_baseline", False):
        return "Baseline"
    if getattr(cfg, "do_test", False):
        return "KVComm"
    return "KVComm"


def generate_run_name(cfg) -> str:
    model_A_short = get_model_short_name(cfg.model_A)
    model_B_short = get_model_short_name(cfg.model_B)

    base_name = f"{model_A_short}-to-{model_B_short}"

    if cfg.top_layers > 0:
        layer_info = f"top{cfg.top_layers}"
    elif cfg.layers_list != [-1]:
        layer_info = f"layers{cfg.layers_list}"
    else:
        layer_to_str = "ALL" if cfg.layer_to < 0 else cfg.layer_to
        layer_info = f"from{cfg.layer_from}to{layer_to_str}"
    base_name += f"_{layer_info}"

    method = get_method_name(cfg)
    base_name += f"_{method}"

    return base_name


def generate_run_name_multi_agent(cfg) -> str:
    model_A1_short = get_model_short_name(cfg.model_A1)
    model_A2_short = get_model_short_name(cfg.model_A2)
    model_B_short = get_model_short_name(cfg.model_B)

    base_name = f"{model_A1_short}-{model_A2_short}-to-{model_B_short}"

    if cfg.top_layers > 0:
        layer_info = f"top{cfg.top_layers}"
    elif cfg.layers_list != [-1]:
        layer_info = f"layers{cfg.layers_list}"
    else:
        layer_to_str = "ALL" if cfg.layer_to < 0 else cfg.layer_to
        layer_info = f"from{cfg.layer_from}to{layer_to_str}"
    base_name += f"_{layer_info}"

    method = get_method_name(cfg)
    base_name += f"_{method}"

    return base_name