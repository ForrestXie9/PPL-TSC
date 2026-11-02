#!/usr/bin/env python3
"""Save the user's API connection settings and key locally."""

from __future__ import annotations

import argparse
from getpass import getpass
import json
import math
import os
from pathlib import Path
import shlex
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
ENV_PATH = ROOT / ".env"
PROFILE_PATH = ROOT / "configs/api.local.json"
PROFILE_FIELDS = {"endpoint", "request_path", "model", "api_key_env", "temperature", "auth_mode"}


def _read_env_key():
    key = os.getenv("LLM_API_KEY", "").strip()
    if key or not ENV_PATH.is_file():
        return key
    for line in ENV_PATH.read_text().splitlines():
        name, separator, value = line.strip().removeprefix("export ").partition("=")
        if separator and name.strip() == "LLM_API_KEY":
            parts = shlex.split(value, comments=True)
            return parts[0] if parts else ""
    return ""


def _read_profile():
    if not PROFILE_PATH.is_file():
        return {}
    profile = json.loads(PROFILE_PATH.read_text())
    if not isinstance(profile, dict):
        raise ValueError("API settings must be a JSON object")
    return {key: value for key, value in profile.items() if key in PROFILE_FIELDS}


def _write_private(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _save(key, profile):
    lines = ENV_PATH.read_text().splitlines() if ENV_PATH.is_file() else []
    lines = [line for line in lines if line.strip().removeprefix("export ").partition("=")[0].strip() != "LLM_API_KEY"]
    lines.append("export LLM_API_KEY=" + shlex.quote(key))
    _write_private(ENV_PATH, "\n".join(lines) + "\n")
    _write_private(PROFILE_PATH, json.dumps(profile, indent=2, allow_nan=False) + "\n")


def _prompt(label, default):
    suffix = f" [{default}]" if default else ""
    return input(label + suffix + ": ").strip() or default


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", help="API hostname or base URL")
    parser.add_argument("--model", help="Exact model name accepted by your API")
    parser.add_argument("--request-path", help="Chat request path (default: /v1/chat/completions)")
    parser.add_argument("--temperature", type=float, help="Sampling temperature (default: 0.8)")
    parser.add_argument("--auth-mode", choices=["bearer", "api_key"], help="Authentication header (default: bearer)")
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--test", action="store_true", help="Send one short test request after entering settings")
    parser.add_argument("--show", action="store_true", help="Show saved settings and whether a key is present")
    args = parser.parse_args()
    existing = _read_profile()
    if args.show:
        print(json.dumps(existing, indent=2))
        print("API key: " + ("configured" if _read_env_key() else "missing"))
        return 0

    endpoint = args.endpoint or str(existing.get("endpoint", ""))
    model = args.model or str(existing.get("model", ""))
    if not args.non_interactive:
        if not args.endpoint:
            endpoint = _prompt("API address", endpoint)
        if not args.model:
            model = _prompt("Model name", model)
    from utilities.utils import normalize_llm_api_host
    endpoint = normalize_llm_api_host(endpoint)
    model = model.strip()
    if not endpoint or not model:
        parser.error("Enter an API address and model name, or pass --endpoint and --model")
    path = args.request_path or str(existing.get("request_path", "/v1/chat/completions"))
    if not path.startswith("/") or path.startswith("//") or any(c in path for c in "\r\n?#"):
        parser.error("The chat request path must start with / and contain no query, fragment or newline")
    temperature = args.temperature if args.temperature is not None else float(existing.get("temperature", 0.8))
    if not math.isfinite(temperature) or not 0 <= temperature <= 2:
        parser.error("Temperature must be between 0 and 2")
    auth = args.auth_mode or str(existing.get("auth_mode", "bearer")).replace("-", "_")
    if auth not in {"bearer", "api_key"}:
        parser.error("Authentication must be bearer or api_key")
    key = _read_env_key()
    if not key and not args.non_interactive:
        key = getpass("API key (input hidden; leave empty for a local API without authentication): ").strip()
    from utilities.utils import is_local_api_host
    if not key and not is_local_api_host(endpoint):
        parser.error("Set LLM_API_KEY or run interactively to enter a key")

    profile = dict(endpoint=endpoint, model=model, request_path=path,
                   api_key_env="LLM_API_KEY", temperature=temperature, auth_mode=auth)
    if args.test:
        from ppl_tsc.llm.client import GeneralAPIInterface
        reply = GeneralAPIInterface(endpoint, key, model, auth_mode=auth,
                                    request_path=path, temperature=temperature).get_response("Reply with OK only.", max_retries=1)
        print("API test succeeded: " + (reply.replace(key, "[redacted]") if key else reply)[:80])
    _save(key, profile)
    print("Saved API settings locally. Run python run.py --config CONFIG.json to distill.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
