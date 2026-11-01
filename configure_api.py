#!/usr/bin/env python3
"""Configure and test the local LLM API profile without exposing its key."""

from __future__ import annotations

import argparse
from getpass import getpass
import json
import os
from pathlib import Path
import shlex
import ssl
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"
PROFILE_PATH = ROOT / "configs/api.local.json"
PROVIDERS = {
    "yunwu": {
        "endpoint": "yunwu.ai",
        "request_path": "/v1/chat/completions",
        "model": "gpt-4o-mini",
        "auth_mode": "bearer",
    },
    "openai_compatible": {
        "endpoint": "api.n1n.ai",
        "request_path": "/v1/chat/completions",
        "model": "gemini-3.1-flash-lite",
        "temperature": 0.8,
        "auth_mode": "bearer",
    },
    "local": {
        "endpoint": "127.0.0.1:8000",
        "request_path": "/v1/chat/completions",
        "model": "qwen2.5-32b",
        "auth_mode": "bearer",
    },
}
PREFERRED_MODELS = (
    "gemini-3.1-flash-lite",
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "gpt-4o",
    "qwen2.5-32b",
)


def _read_env_key() -> str:
    process_value = os.getenv("LLM_API_KEY", "").strip()
    if process_value:
        return process_value
    if not ENV_PATH.is_file():
        return ""
    for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, separator, raw_value = line.partition("=")
        if separator and key.strip() == "LLM_API_KEY":
            values = shlex.split(raw_value, comments=True, posix=True)
            return values[0].strip() if values else ""
    return ""


def _read_profile() -> dict[str, Any]:
    if not PROFILE_PATH.is_file():
        return {}
    with PROFILE_PATH.open(encoding="utf-8") as f:
        value = json.load(f)
    return value if isinstance(value, dict) else {}


def _origin(endpoint: str) -> str:
    raw = endpoint.strip().rstrip("/")
    if not raw:
        raise ValueError("API endpoint cannot be empty")
    if "://" not in raw:
        local = raw.startswith(("127.", "localhost", "0.0.0.0", "[::1]"))
        raw = f"{'http' if local else 'https'}://{raw}"
    parsed = urlparse(raw)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid API endpoint: {endpoint!r}")
    return f"{parsed.scheme}://{parsed.netloc}"


def _headers(api_key: str, auth_mode: str) -> dict[str, str]:
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if not api_key or api_key == "EMPTY":
        return headers
    if auth_mode == "api-key":
        headers["api-key"] = api_key
    else:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _request_json(
    *,
    endpoint: str,
    path: str,
    api_key: str,
    auth_mode: str,
    payload: dict[str, Any] | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    url = f"{_origin(endpoint)}{path if path.startswith('/') else '/' + path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(
        url,
        data=data,
        headers=_headers(api_key, auth_mode),
        method="POST" if payload is not None else "GET",
    )
    try:
        with urlopen(
            request,
            timeout=timeout,
            context=ssl.create_default_context(),
        ) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Cannot connect to {url}: {exc.reason}") from exc
    value = json.loads(body)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object from {url}")
    return value


def list_models(endpoint: str, api_key: str, auth_mode: str) -> list[str]:
    value = _request_json(
        endpoint=endpoint,
        path="/v1/models",
        api_key=api_key,
        auth_mode=auth_mode,
        timeout=30,
    )
    models = value.get("data", [])
    return sorted(
        {
            str(item["id"])
            for item in models
            if isinstance(item, dict) and item.get("id")
        },
        key=str.casefold,
    )


def test_chat(
    endpoint: str,
    request_path: str,
    model: str,
    api_key: str,
    auth_mode: str,
) -> str:
    value = _request_json(
        endpoint=endpoint,
        path=request_path,
        api_key=api_key,
        auth_mode=auth_mode,
        payload={
            "model": model,
            "messages": [{"role": "user", "content": "Reply with OK only."}],
            "temperature": 0,
            "max_tokens": 8,
        },
    )
    try:
        return str(value["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Chat response has no choices[0].message.content") from exc


def _prompt(label: str, default: str) -> str:
    value = input(f"{label} [{default}]: ").strip()
    return value or default


def _choose_provider(default: str) -> str:
    choices = ("openai_compatible", "yunwu", "local")
    print("API backend:")
    for index, name in enumerate(choices, start=1):
        marker = " (default)" if name == default else ""
        print(f"  {index}. {name}{marker}")
    value = input("Select a number or name: ").strip()
    if not value:
        return default
    if value.isdigit() and 1 <= int(value) <= len(choices):
        return choices[int(value) - 1]
    if value in choices:
        return value
    raise ValueError(f"Unknown provider: {value}")


def _choose_model(models: list[str], default: str) -> str:
    ordered = [name for name in PREFERRED_MODELS if name in models]
    ordered.extend(name for name in models if name not in ordered)
    shown = ordered[:30]
    print(f"The API returned {len(models)} models; showing the first {len(shown)}:")
    for index, name in enumerate(shown, start=1):
        marker = " (recommended)" if name == default else ""
        print(f"  {index:2d}. {name}{marker}")
    value = input(f"Model number or full name [{default}]: ").strip()
    if not value:
        return default
    if value.isdigit() and 1 <= int(value) <= len(shown):
        return shown[int(value) - 1]
    return value


def _save(api_key: str, profile: dict[str, Any]) -> None:
    ENV_PATH.write_text(
        "# Local secret generated by configure_api.py; do not commit.\n"
        f"export LLM_API_KEY={shlex.quote(api_key)}\n",
        encoding="utf-8",
    )
    ENV_PATH.chmod(0o600)
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _show() -> int:
    profile = _read_profile()
    key = _read_env_key()
    print(f"Secret file: {ENV_PATH} ({'configured' if key else 'missing'})")
    print(f"API profile: {PROFILE_PATH} ({'configured' if profile else 'missing'})")
    if profile:
        print(json.dumps(profile, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Interactively configure API credentials and model selection."
    )
    parser.add_argument("--provider", choices=sorted(PROVIDERS))
    parser.add_argument("--endpoint")
    parser.add_argument("--request-path")
    parser.add_argument("--model")
    parser.add_argument("--temperature", type=float, help="Preserve the existing temperature unless overridden")
    parser.add_argument("--auth-mode", choices=("bearer", "api-key"))
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--no-list-models", action="store_true")
    parser.add_argument("--no-test", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.temperature is not None and not 0 <= args.temperature <= 2:
        parser.error("--temperature must be between 0 and 2")

    if args.show:
        return _show()

    existing = _read_profile()
    default_provider = str(existing.get("provider", "openai_compatible"))
    provider = (
        args.provider
        or (default_provider if args.non_interactive else _choose_provider(default_provider))
    )
    defaults = dict(PROVIDERS[provider])
    if existing.get("provider") == provider:
        defaults.update(existing)
    endpoint = args.endpoint or (
        str(defaults["endpoint"])
        if args.non_interactive
        else _prompt("API hostname", str(defaults["endpoint"]))
    )
    request_path = args.request_path or (
        str(defaults["request_path"])
        if args.non_interactive
        else _prompt("Chat request path", str(defaults["request_path"]))
    )
    auth_mode = args.auth_mode or str(defaults.get("auth_mode", "bearer"))

    api_key = _read_env_key()
    if provider == "local" and not api_key:
        api_key = "EMPTY"
    elif not api_key and not args.non_interactive:
        api_key = getpass("API key (input hidden): ").strip()
    if not api_key:
        raise ValueError(
            "No API key found. Set LLM_API_KEY or run interactively to enter one."
        )

    model = args.model or str(defaults["model"])
    if not args.no_list_models:
        try:
            models = list_models(endpoint, api_key, auth_mode)
            if models:
                if args.non_interactive:
                    if model not in models:
                        print(
                            f"Warning: selected model {model!r} was not listed by the API.",
                            file=sys.stderr,
                        )
                else:
                    model = _choose_model(models, model)
        except Exception as exc:
            if args.non_interactive:
                raise
            print(f"Could not fetch models; enter the model manually: {exc}")
            model = _prompt("Model name", model)
    elif not args.non_interactive and not args.model:
        model = _prompt("Model name", model)

    if not args.no_test:
        reply = test_chat(endpoint, request_path, model, api_key, auth_mode)
        print(f"API test succeeded; model reply: {reply[:80]}")

    profile = {
        "provider": provider,
        "endpoint": endpoint,
        "request_path": request_path,
        "model": model,
        "api_key_env": "LLM_API_KEY",
        "temperature": args.temperature if args.temperature is not None else float(defaults.get("temperature", 0.0)),
        "auth_mode": auth_mode,
    }
    _save(api_key, profile)
    print(f"API key saved locally to: {ENV_PATH}")
    print(f"Model settings saved to: {PROFILE_PATH}")
    print("Run python run.py ... directly; sourcing .env is not required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
