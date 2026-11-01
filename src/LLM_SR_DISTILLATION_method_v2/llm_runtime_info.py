"""Display the configured LLM host, model, backend and callable at startup."""
from __future__ import annotations

import os


def print_llm_startup_banner(
    llm_api_endpoint: str,
    llm_model: str,
    *,
    title: str = "LLM configuration (distillation)",
    cli_llm_model: str | None = None,
) -> None:
    import LLM_SR_DISTILLATION.llm_generator as _lg

    mod = getattr(_lg.run_llm, "__module__", "")
    name = getattr(_lg.run_llm, "__name__", "run_llm")
    api_ver = os.getenv("LLM_API_VERSION", "2024-10-21")

    backend_env = (os.getenv("LLM_BACKEND") or "").strip().lower()
    is_comp = backend_env in ("comp", "comp_openai", "azure_apim", "azure") or (
        "comp_openai_client" in mod
    )
    is_jiekou = backend_env in ("jiekou", "jiekou_ai")
    is_yunwu = backend_env in ("yunwu", "yunwu_ai")

    if is_comp:
        path_hint = (
            f"/azure/openai/deployments/<{llm_model}>/chat/completions"
            f"?api-version={api_ver}"
        )
    elif is_jiekou:
        path_hint = os.getenv("LLM_REQUEST_PATH", "/openai/v1/chat/completions")
    elif is_yunwu:
        path_hint = os.getenv("LLM_REQUEST_PATH", "/v1/chat/completions")
    else:
        path_hint = os.getenv("LLM_REQUEST_PATH", "/v1/chat/completions")

    env_llm_model = os.getenv("LLM_MODEL")
    if is_comp:
        backend = "COMP OpenAI (external comp_openai_client, LLM_BACKEND=comp)"
    elif is_jiekou:
        backend = f"Jiekou.AI OpenAI-compatible (GeneralAPIInterface), module {mod}"
    elif is_yunwu:
        backend = f"Yunwu OpenAI-compatible (GeneralAPIInterface), module {mod}"
    else:
        backend = f"General OpenAI-compatible (agent.llm_integration), module {mod}"

    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)
    print(f"  LLM_BACKEND:                                         {backend_env or '(unset; general OpenAI-compatible)'}")
    print(f"  Gateway host (LLM_API_ENDPOINT / --llm_api_endpoint): {llm_api_endpoint}")
    cli_m = (cli_llm_model or "").strip()
    print(f"  CLI --llm_model:                                    {cli_m or '(not supplied)'}"
          f"{' (takes priority)' if cli_m else ''}")
    print(f"  Environment LLM_MODEL:                              {env_llm_model or '(unset)'}")
    if is_comp:
        print(
            "  Deployment precedence (COMP):                       "
            "--llm_model > nonempty embedded COMP_MODEL > LLM_MODEL > gpt4o"
        )
    elif is_jiekou:
        print(
            "  Model precedence (Jiekou):                           "
            "--llm_model > nonempty embedded JIEKOU_MODEL > LLM_MODEL > gpt-4o-mini"
        )
    elif is_yunwu:
        print(
            "  Model precedence (Yunwu):                            "
            "--llm_model > nonempty embedded YUNWU_MODEL > LLM_MODEL > gpt-4o-mini"
        )
    else:
        print(
            "  Model precedence (general):                          "
            "--llm_model > nonempty embedded GENERAL_MODEL > LLM_MODEL > gpt-3.5-turbo"
        )
    print(f"  Selected model/deployment:                          {llm_model}")
    print(f"  API version (LLM_API_VERSION):                       {api_ver if is_comp else '(COMP only; generally unused by other gateways)'}")
    print(f"  Request path:                                       {path_hint}")
    print(f"  run_llm callable:                                   {mod}.{name}")
    print(f"  Backend:                                            {backend}")
    print("=" * 64 + "\n")
