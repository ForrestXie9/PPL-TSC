import os
import re
import torch
import random
import numpy as np
import datetime
import logging
import json
import copy
from typing import Optional


def infer_dataset_name_from_teacher_path(teacher_data_path: Optional[str]) -> Optional[str]:
    """
    Parse paths like .../<Dataset>/flowK.json, flowK.jsonl, or flowK.jsonl.gz so CityFlow assets
    resolve to project_root/data/<Dataset>/ (matches collect_teacher_data layouts).
    """
    if not teacher_data_path:
        return None
    try:
        norm = os.path.abspath(teacher_data_path).replace("\\", "/")
    except Exception:
        norm = str(teacher_data_path).replace("\\", "/")
    m = re.search(r"/([^/]+)/flow\d+\.(?:jsonl|json)(?:\.gz)?$", norm)
    if m:
        return m.group(1)
    return None


def normalize_llm_api_host(api_endpoint: Optional[str]) -> str:
    """Strip scheme/path so http.client HTTPSConnection receives hostname only."""
    s = (api_endpoint or "").strip()
    if not s:
        return s
    s = re.sub(r"^https?://", "", s, flags=re.I)
    s = s.split("/", 1)[0].strip()
    return s


def resolve_llm_connection_settings(
    args_endpoint: Optional[str], args_key: Optional[str], args_model: Optional[str]
):
    """Resolve CLI values, then environment values, then gateway defaults."""
    backend = (os.getenv("LLM_BACKEND") or "").strip().lower()
    if backend in {"comp", "comp_openai", "azure_apim", "azure"}:
        raise ValueError("The private COMP/Azure adapter is not included; use an OpenAI-compatible provider")
    if backend in {"jiekou", "jiekou_ai"}:
        default_endpoint, default_model = "api.jiekou.ai", "gpt-4o-mini"
    elif backend in {"yunwu", "yunwu_ai"}:
        default_endpoint, default_model = "yunwu.ai", "gpt-4o-mini"
    else:
        default_endpoint, default_model = "127.0.0.1:8000", "qwen2.5-32b"
    endpoint = normalize_llm_api_host(args_endpoint or os.getenv("LLM_API_ENDPOINT") or default_endpoint)
    key = (args_key or os.getenv("LLM_API_KEY") or "").strip()
    model = (args_model or os.getenv("LLM_MODEL") or default_model).strip()
    return endpoint, key, model


def llm_api_key_missing_message() -> str:
    """Describe the supported credential configuration without referencing source profiles."""
    return "LLM API key missing. Run configure_api.py, set LLM_API_KEY, or pass --llm_api_key."


def apply_llm_backend_cli_choice(cli_choice: Optional[str]) -> None:
    """Select a public gateway preset while preserving explicit environment settings."""
    if cli_choice is None:
        return
    choice = str(cli_choice).strip()
    if choice == "2":
        raise ValueError("Backend 2 requires an unbundled private adapter; use backend 1 instead")
    if choice not in {"1", "3", "4"}:
        raise ValueError("LLM backend must be 1 (general), 3 (Jiekou), or 4 (Yunwu)")
    if choice == "1":
        os.environ.pop("LLM_BACKEND", None)
    else:
        os.environ["LLM_BACKEND"] = "jiekou" if choice == "3" else "yunwu"
    request_path = "/openai/v1/chat/completions" if choice == "3" else "/v1/chat/completions"
    os.environ.setdefault("LLM_REQUEST_PATH", request_path)
    os.environ.setdefault("LLM_AUTH_MODE", "bearer")
    if choice in {"1", "4"}:
        os.environ.setdefault("LLM_TEMPERATURE", "0")


def validate_cityflow_roadnet_file(data_dir: str, roadnet_file: str = "roadnet.json") -> None:
    """
    Ensure roadnet JSON exists and is non-empty parseable JSON (fail fast with a clear path).
    """
    if not data_dir:
        raise ValueError("CityFlow data_dir is empty")
    road_path = os.path.join(data_dir, roadnet_file)
    if not os.path.isfile(road_path):
        raise FileNotFoundError(
            f"roadnet file not found: {road_path}. "
            f"Set meta.dataset in teacher data, fix teacher_data_path, or add {roadnet_file} under {data_dir}."
        )
    with open(road_path, encoding="utf-8") as f:
        raw = f.read()
    if not raw.strip():
        raise ValueError(
            f"roadnet file is empty or whitespace-only: {road_path} "
            f"(size={os.path.getsize(road_path)} bytes). Replace with a valid CityFlow roadnet.json."
        )
    try:
        json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"roadnet file is not valid JSON: {road_path}: {e}") from e


def get_config(file_path: str = "config.json") -> dict:
    """Load a traffic configuration from an explicit, reproducible path."""
    with open(file_path) as f:
        config = json.load(f)
    config['now'] = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    config['logger'] = None
    return config


def get_agent(name):
    import agent
    try:
        return getattr(agent, name)
    except AttributeError as exc:
        raise ValueError(f"Unsupported bundled agent: {name}") from exc


def set_logger(config):
    if config['save_result']:
        # use 'results' directory instead of 'log'
        config['log_path'] = 'results/{}/{}'.format(config['inter_name'], config['cur_agent'])
        # ensure the directory exists before attempting to write
        make_dir(config['log_path'])
        if config['flow_idx'] == 0:
            with open(os.path.join(config['log_path'], 'config.json'), 'w') as fout:
                json.dump(config, fout)

    # Use root logger, but make handler attachment idempotent.
    # On Windows multiprocessing / repeated runs, root logger can already have handlers
    # (e.g., from previous experiments or libraries). If we keep adding, log lines duplicate.
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s -%(message)s'))
    logger.addHandler(sh)
    if config['save_result']:
        fh = logging.FileHandler(
            os.path.join(
                config['log_path'],
                '{}_{}_{}.log'.format(config['cur_agent'], config['flow_idx'], config['now'])
            )
        )
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s -%(message)s'))
        logger.addHandler(fh)
    config['logger'] = logger


def release_logger(config):
    if config['logger'] is not None:
        for handler in config['logger'].handlers[:]:
            config['logger'].removeHandler(handler)


def set_seed(seed, use_torch=True):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if not use_torch:
        return
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)


def set_thread(num_thread=None):
    if num_thread is not None:
        torch.set_num_threads(num_thread)
    else:
        torch.set_num_threads(1)
        # import platform
        # if platform.system() == 'Linux':  # Darwin
        #     torch.set_num_threads(1)


def copy_model_params(source_model, target_model):
    for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
        target_param.data.copy_(source_param.clone())


def _is_dumpable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False


def get_dumpable_config(config: dict):
    dumpable_config = {}
    for k, v in config.items():
        if _is_dumpable({k: v}):
            dumpable_config[k] = copy.deepcopy(v)
    return dumpable_config


def list_with_unique_element(original_list):  # Preserve order while removing duplicates.
    new_list = []
    for elem in original_list:
        if elem not in new_list:
            new_list.append(elem)
    return new_list


def make_dir(dir_name):
    """Create directory if it doesn't exist. If it exists, simply use it (don't rename)."""
    os.makedirs(dir_name, exist_ok=True)
