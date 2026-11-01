"""启动时打印 LLM 网关 / 部署名 / run_llm 绑定，便于确认是否使用 COMP「新的接口」。"""
from __future__ import annotations

import os


def print_llm_startup_banner(
    llm_api_endpoint: str,
    llm_model: str,
    *,
    title: str = "LLM 调用（蒸馏）",
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
        backend = "COMP OpenAI（新的接口 comp_openai_client，LLM_BACKEND=comp）"
    elif is_jiekou:
        backend = f"Jiekou.AI OpenAI 兼容（GeneralAPIInterface），模块 {mod}"
    elif is_yunwu:
        backend = f"云雾 OpenAI 兼容（GeneralAPIInterface），模块 {mod}"
    else:
        backend = f"通用 OpenAI 兼容（agent.llm_integration），模块 {mod}"

    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)
    print(f"  LLM_BACKEND:                                         {backend_env or '(未设置 → 通用 OpenAI 兼容)'}")
    print(f"  网关主机 (LLM_API_ENDPOINT / --llm_api_endpoint): {llm_api_endpoint}")
    cli_m = (cli_llm_model or "").strip()
    print(f"  命令行 --llm_model:                                {cli_m or '(未传)'}"
          f"{' → 优先生效' if cli_m else ''}")
    print(f"  环境变量 LLM_MODEL:                                {env_llm_model or '(未设置)'}")
    if is_comp:
        print(
            "  部署名解析 (COMP):                                  "
            "--llm_model > 嵌入式 COMP_MODEL（若存在且非空；否则再读 LLM_MODEL）> gpt4o"
        )
    elif is_jiekou:
        print(
            "  模型名解析 (Jiekou):                                "
            "--llm_model > 嵌入式 JIEKOU_MODEL（若非空）> LLM_MODEL > gpt-4o-mini"
        )
    elif is_yunwu:
        print(
            "  模型名解析 (云雾):                                  "
            "--llm_model > 嵌入式 YUNWU_MODEL（若非空）> LLM_MODEL > gpt-4o-mini"
        )
    else:
        print(
            "  模型名解析 (通用):                                  "
            "--llm_model > 嵌入式 GENERAL_MODEL（若非空）> LLM_MODEL > gpt-3.5-turbo"
        )
    print(f"  实际采用的模型/部署名:                              {llm_model}")
    print(f"  API 版本 (LLM_API_VERSION):                        {api_ver if is_comp else '(通用网关通常忽略，仅 COMP 使用)'}")
    print(f"  请求路径示意:                                        {path_hint}")
    print(f"  run_llm 绑定:                                        {mod}.{name}")
    print(f"  接口类型:                                            {backend}")
    print("=" * 64 + "\n")
