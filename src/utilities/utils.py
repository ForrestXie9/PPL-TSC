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
    args_endpoint: Optional[str],
    args_key: Optional[str],
    args_model: Optional[str],
):
    """
    Merge CLI with env. Argparse should use default=None so LLM_API_ENDPOINT / LLM_MODEL are honored.

    **Embedded profiles** (optional modules under ``LLM_SR_DISTILLATION_method_v2``):

    - ``--llm_backend 2`` / ``LLM_BACKEND=comp``: ``comp_llm_embedded`` → ``COMP_*`` constants.
    - ``--llm_backend 3`` / ``LLM_BACKEND=jiekou``: ``jiekou_llm_embedded`` → ``JIEKOU_*`` constants.
    - ``--llm_backend 4`` / ``LLM_BACKEND=yunwu``: ``yunwu_llm_embedded`` → ``YUNWU_*`` constants.
    - ``--llm_backend 1`` / general: ``general_llm_embedded`` → ``GENERAL_*`` constants.

    **Endpoint / key**: ``CLI > LLM_* env > embedded_* > built-in default host``.

    **Model / deployment name**:

    - ``--llm_model`` always wins if provided.

    **COMP**: non-empty embedded ``COMP_MODEL`` is used **before** ``LLM_MODEL`` (avoids stale global env).

    **General / Jiekou**: non-empty embedded ``GENERAL_MODEL`` / ``JIEKOU_MODEL`` before ``LLM_MODEL``;
    fallback ``gpt-3.5-turbo`` (general) or ``gpt-4o-mini`` (Jiekou)。
    """
    backend = (os.getenv("LLM_BACKEND") or "").strip().lower()
    is_comp = backend in ("comp", "comp_openai", "azure_apim", "azure")
    is_jiekou = backend in ("jiekou", "jiekou_ai")
    is_yunwu = backend in ("yunwu", "yunwu_ai")

    embedded = None
    if is_comp:
        try:
            from LLM_SR_DISTILLATION_method_v2 import comp_llm_embedded as embedded
        except ImportError:
            embedded = None

    jk_emb_mod = None
    if is_jiekou:
        try:
            from LLM_SR_DISTILLATION_method_v2 import jiekou_llm_embedded as jk_emb_mod
        except ImportError:
            jk_emb_mod = None

    yw_emb_mod = None
    if is_yunwu:
        try:
            from LLM_SR_DISTILLATION_method_v2 import yunwu_llm_embedded as yw_emb_mod
        except ImportError:
            yw_emb_mod = None

    gen_emb = None
    if not is_comp and not is_jiekou and not is_yunwu:
        try:
            from LLM_SR_DISTILLATION_method_v2 import general_llm_embedded as gen_emb
        except ImportError:
            gen_emb = None

    emb_ep = ""
    emb_key = ""
    emb_model = ""
    if embedded is not None:
        emb_ep = (getattr(embedded, "COMP_API_ENDPOINT", "") or "").strip()
        emb_key = (getattr(embedded, "COMP_API_KEY", "") or "").strip()
        emb_model = (getattr(embedded, "COMP_MODEL", "") or "").strip()

    jk_ep = ""
    jk_key = ""
    jk_model = ""
    if jk_emb_mod is not None:
        jk_ep = (getattr(jk_emb_mod, "JIEKOU_API_ENDPOINT", "") or "").strip()
        jk_key = (getattr(jk_emb_mod, "JIEKOU_API_KEY", "") or "").strip()
        jk_model = (getattr(jk_emb_mod, "JIEKOU_MODEL", "") or "").strip()

    yw_ep = ""
    yw_key = ""
    yw_model = ""
    if yw_emb_mod is not None:
        yw_ep = (getattr(yw_emb_mod, "YUNWU_API_ENDPOINT", "") or "").strip()
        yw_key = (getattr(yw_emb_mod, "YUNWU_API_KEY", "") or "").strip()
        yw_model = (getattr(yw_emb_mod, "YUNWU_MODEL", "") or "").strip()

    gen_ep = ""
    gen_key = ""
    gen_model = ""
    if gen_emb is not None:
        gen_ep = (getattr(gen_emb, "GENERAL_API_ENDPOINT", "") or "").strip()
        gen_key = (getattr(gen_emb, "GENERAL_API_KEY", "") or "").strip()
        gen_model = (getattr(gen_emb, "GENERAL_MODEL", "") or "").strip()

    ep_cli_or_env = (args_endpoint if args_endpoint else (os.getenv("LLM_API_ENDPOINT") or "")).strip()
    if is_comp:
        endpoint = normalize_llm_api_host(
            ep_cli_or_env or emb_ep or "comp.azure-api.net"
        )
    elif is_jiekou:
        endpoint = normalize_llm_api_host(ep_cli_or_env or jk_ep or "api.jiekou.ai")
    elif is_yunwu:
        endpoint = normalize_llm_api_host(ep_cli_or_env or yw_ep or "yunwu.ai")
    else:
        endpoint = normalize_llm_api_host(ep_cli_or_env or gen_ep or "api.bltcy.ai")

    key_raw = args_key if args_key else os.getenv("LLM_API_KEY", "")
    key = (key_raw or "").strip()
    if not key:
        if is_comp:
            key = emb_key
        elif is_jiekou:
            key = jk_key
        elif is_yunwu:
            key = yw_key
        else:
            key = gen_key

    # Model: CLI > embedded (per backend, if non-empty) > LLM_MODEL > fallback
    if args_model:
        model = args_model.strip()
    elif is_comp and emb_model:
        model = emb_model
    elif is_jiekou and jk_model:
        model = jk_model
    elif is_yunwu and yw_model:
        model = yw_model
    elif (not is_comp) and (not is_jiekou) and (not is_yunwu) and gen_model:
        model = gen_model
    else:
        model = os.getenv("LLM_MODEL") or ""

    if not model:
        if is_comp:
            model = "gpt4o"
        elif is_jiekou:
            model = "gpt-4o-mini"
        elif is_yunwu:
            model = "gpt-4o-mini"
        else:
            model = "gpt-3.5-turbo"

    return endpoint, key, model


def llm_api_key_missing_message() -> str:
    """
    Short hint for ValueError when no API key was resolved (depends on ``LLM_BACKEND`` env).
    Call **after** ``apply_llm_backend_cli_choice`` so COMP vs general is accurate.
    """
    backend = (os.getenv("LLM_BACKEND") or "").strip().lower()
    if backend in ("comp", "comp_openai", "azure_apim", "azure"):
        return (
            "LLM API key 未配置。请任选其一：环境变量 LLM_API_KEY；命令行 --llm_api_key；"
            "或在 LLM_SR_DISTILLATION_method_v2/comp_llm_embedded.py 填写 COMP_API_KEY。"
        )
    if backend in ("jiekou", "jiekou_ai"):
        return (
            "LLM API key 未配置。请任选其一：环境变量 LLM_API_KEY；命令行 --llm_api_key；"
            "或在 LLM_SR_DISTILLATION_method_v2/jiekou_llm_embedded.py 填写 JIEKOU_API_KEY。"
        )
    if backend in ("yunwu", "yunwu_ai"):
        return (
            "LLM API key 未配置。请任选其一：环境变量 LLM_API_KEY；命令行 --llm_api_key；"
            "或在 LLM_SR_DISTILLATION_method_v2/yunwu_llm_embedded.py 填写 YUNWU_API_KEY。"
        )
    return (
        "LLM API key 未配置。请任选其一：环境变量 LLM_API_KEY；命令行 --llm_api_key；"
        "或在 LLM_SR_DISTILLATION_method_v2/general_llm_embedded.py 填写 GENERAL_API_KEY。"
    )


def apply_llm_backend_cli_choice(cli_choice: Optional[str]) -> None:
    """
    Map --llm_backend 1|2|3|4 into process env so ``run_llm`` / ``resolve_llm_connection_settings`` see it.

    - ``1``: general OpenAI-compatible (drops LLM_BACKEND). Optionally loads
      ``general_llm_embedded`` defaults into ``LLM_REQUEST_PATH`` / ``LLM_AUTH_MODE`` / ``LLM_TEMPERATURE``
      via ``setdefault`` when those env vars are unset.
    - ``2``: COMP (sets LLM_BACKEND=comp).
    - ``3``: Jiekou.AI OpenAI-compatible (sets LLM_BACKEND=jiekou; same ``run_llm`` path as ``1``,
      default host ``api.jiekou.ai`` and path ``/openai/v1/chat/completions`` from ``jiekou_llm_embedded``).
    - ``4``: Yunwu OpenAI-compatible (sets LLM_BACKEND=yunwu; default host ``yunwu.ai`` and
      path ``/v1/chat/completions`` from ``yunwu_llm_embedded``).
    Default CLI value is ``1``：不写命令行时也为第一种，shell 里的 LLM_BACKEND 会被清除。
    """
    if cli_choice is None:
        return
    c = str(cli_choice).strip()
    if c == "2":
        os.environ["LLM_BACKEND"] = "comp"
        try:
            from LLM_SR_DISTILLATION_method_v2 import comp_llm_embedded as _emb

            v = (getattr(_emb, "COMP_API_VERSION", "") or "").strip()
            if v:
                os.environ.setdefault("LLM_API_VERSION", v)
            t = (getattr(_emb, "COMP_TEMPERATURE", "") or "").strip()
            if t:
                os.environ.setdefault("LLM_TEMPERATURE", t)
        except ImportError:
            pass
    elif c == "3":
        os.environ["LLM_BACKEND"] = "jiekou"
        try:
            from LLM_SR_DISTILLATION_method_v2 import jiekou_llm_embedded as _jk

            rp = (getattr(_jk, "JIEKOU_REQUEST_PATH", "") or "").strip()
            if rp:
                os.environ.setdefault("LLM_REQUEST_PATH", rp)
            am = (getattr(_jk, "JIEKOU_AUTH_MODE", "") or "").strip()
            if am:
                os.environ.setdefault("LLM_AUTH_MODE", am)
            t = (getattr(_jk, "JIEKOU_TEMPERATURE", "") or "").strip()
            if t:
                os.environ.setdefault("LLM_TEMPERATURE", t)
        except ImportError:
            pass
    elif c == "4":
        os.environ["LLM_BACKEND"] = "yunwu"
        try:
            from LLM_SR_DISTILLATION_method_v2 import yunwu_llm_embedded as _yw

            rp = (getattr(_yw, "YUNWU_REQUEST_PATH", "") or "").strip()
            if rp:
                os.environ.setdefault("LLM_REQUEST_PATH", rp)
            am = (getattr(_yw, "YUNWU_AUTH_MODE", "") or "").strip()
            if am:
                os.environ.setdefault("LLM_AUTH_MODE", am)
            t = (getattr(_yw, "YUNWU_TEMPERATURE", "") or "").strip()
            if t:
                os.environ.setdefault("LLM_TEMPERATURE", t)
        except ImportError:
            pass
    elif c == "1":
        os.environ.pop("LLM_BACKEND", None)
        try:
            from LLM_SR_DISTILLATION_method_v2 import general_llm_embedded as _g

            rp = (getattr(_g, "GENERAL_REQUEST_PATH", "") or "").strip()
            if rp:
                os.environ.setdefault("LLM_REQUEST_PATH", rp)
            am = (getattr(_g, "GENERAL_AUTH_MODE", "") or "").strip()
            if am:
                os.environ.setdefault("LLM_AUTH_MODE", am)
            t = (getattr(_g, "GENERAL_TEMPERATURE", "") or "").strip()
            if t:
                os.environ.setdefault("LLM_TEMPERATURE", t)
        except ImportError:
            pass


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


def list_with_unique_element(original_list):  # list去重
    new_list = []
    for elem in original_list:
        if elem not in new_list:
            new_list.append(elem)
    return new_list


def make_dir(dir_name):
    """Create directory if it doesn't exist. If it exists, simply use it (don't rename)."""
    os.makedirs(dir_name, exist_ok=True)
