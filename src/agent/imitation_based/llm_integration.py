"""
LLM集成模块，用于调用大语言模型生成符号表达式
支持多种LLM API接口
"""
import importlib
import hashlib
import json
import os
import http.client
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from utilities.utils import normalize_llm_api_host

_CACHE_CONTEXT = "standalone"
_CACHE_CALL_INDEX = 0


def reset_llm_cache_sequence(context: str) -> None:
    """Make each run replayable independently of matrix order or worker count."""
    global _CACHE_CONTEXT, _CACHE_CALL_INDEX
    _CACHE_CONTEXT, _CACHE_CALL_INDEX = context, 0


def _write_request_trace(record: dict, api_key: str) -> None:
    """Archive each attempt for reproducibility, excluding credentials/headers."""
    directory = os.getenv("LLM_TRACE_DIR", "").strip()
    if not directory:
        return
    path = Path(directory).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    # Redact even a provider error that happens to echo the supplied credential.
    def redact(value):
        if isinstance(value, str):
            return value.replace(api_key, "[REDACTED]") if api_key else value
        if isinstance(value, dict):
            return {redact(k): redact(v) for k, v in value.items()}
        if isinstance(value, list):
            return [redact(v) for v in value]
        return value
    target = path / f"{record['request_id']}_attempt{record['attempt']}.json"
    temporary = target.with_suffix(".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(redact(record), stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(target)


def _append_usage_event(event: dict) -> None:
    """Append non-sensitive API accounting metadata when explicitly enabled."""
    raw_path = os.getenv("LLM_USAGE_LOG", "").strip()
    if not raw_path:
        return
    path = Path(raw_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        **event,
    }
    payload = (
        json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n"
    ).encode("utf-8")
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)


def _validate_latin1_header_values(headers: dict) -> None:
    """HTTP headers must be latin-1; fail early with a readable hint."""
    for name, value in headers.items():
        try:
            str(value).encode("latin-1")
        except UnicodeEncodeError as exc:
            preview = str(value)
            if name.lower() == "authorization" and preview.lower().startswith("bearer "):
                preview = preview[:7] + "<api_key>"
            raise ValueError(
                f"HTTP 请求头 {name!r} 含非 ASCII 字符，无法发送。"
                f"常见原因：把文档占位符「你的密钥」原样写进了 LLM_API_KEY / --llm_api_key。"
                f"请改为真实 API Key（仅含英文字母、数字、符号），或 unset LLM_API_KEY 以使用嵌入式密钥。"
                f" 当前值预览: {preview!r}"
            ) from exc


def _open_http_connection(api_endpoint: str, timeout: float = 300.0):
    """Use plain HTTP for local gateways; HTTPS otherwise."""
    host = normalize_llm_api_host(api_endpoint)
    if host.startswith("127.0.0.1") or host.startswith("localhost"):
        return http.client.HTTPConnection(host, timeout=timeout)
    return http.client.HTTPSConnection(host, timeout=timeout)


class LLMInterface:
    """LLM接口基类"""
    
    def __init__(self, api_endpoint: str, api_key: str, model: str, debug_mode: bool = False):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model = model
        self.debug_mode = debug_mode
    
    def get_response(self, prompt: str) -> str:
        """获取LLM响应"""
        raise NotImplementedError


class GeneralAPIInterface(LLMInterface):
    """通用API接口（兼容OpenAI格式）
    
    鉴权方式由 auth_mode 控制：
    - "bearer": Authorization: Bearer <api_key>（常见 OpenAI 兼容网关）
    - "api_key": Api-Key: <api_key>（部分厂商使用独立头名）
    extra_headers 会合并进最终请求头（可传 Accept 等）。
    """
    
    def __init__(
        self,
        api_endpoint: str,
        api_key: str,
        model: str,
        debug_mode: bool = False,
        *,
        auth_mode: str = "bearer",
        extra_headers: Optional[dict] = None,
        request_path: str = "/v1/chat/completions",
        temperature: float = 0.0,
    ):
        super().__init__(api_endpoint, api_key, model, debug_mode)
        self.auth_mode = (auth_mode or "bearer").strip().lower()
        self.extra_headers = dict(extra_headers) if extra_headers else {}
        self.request_path = request_path or "/v1/chat/completions"
        self.temperature = float(temperature)
    
    def _build_headers(self) -> dict:
        headers = {
            "Content-Type": "application/json",
        }
        if self.auth_mode == "api_key":
            headers["Api-Key"] = self.api_key
        else:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)
        return headers
    
    def get_response(self, prompt: str, max_retries: int = 3) -> str:
        """调用通用API接口"""
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.temperature,
            },
            ensure_ascii=False,
        ).encode("utf-8")

        headers = self._build_headers()
        _validate_latin1_header_values(headers)
        request_id = uuid.uuid4().hex

        for attempt in range(max_retries):
            conn = None
            started = time.monotonic()
            trace = {
                "request_id": request_id, "attempt": attempt + 1,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "host": self.api_endpoint, "request_path": self.request_path,
                "request": json.loads(payload), "status": "error",
            }
            try:
                conn = _open_http_connection(self.api_endpoint)
                conn.request("POST", self.request_path, payload, headers)
                res = conn.getresponse()
                raw = res.read()
                text = raw.decode("utf-8", errors="replace")
                status = getattr(res, "status", None) or res.getcode()
                trace.update(http_status=status, response_text=text)

                try:
                    json_data = json.loads(text)
                except json.JSONDecodeError as je:
                    probe = text[:1200].lower()
                    if "<!doctype html" in probe or "<html" in probe:
                        raise RuntimeError(
                            "HTTP 200 但返回网页 HTML（通常是柏拉图/OpenAI 中转商的门户页），不是 Chat API 的 JSON。"
                            "说明 POST 未打到兼容接口。\n"
                            "请按供应商控制台核对：\n"
                            "  • LLM_API_ENDPOINT：应为文档里的 API 主机名（仅域名，不要 https://，不要网页前台域名）；\n"
                            "  • LLM_REQUEST_PATH：一般为 /v1/chat/completions（若文档写明别的路径则按其填写）；\n"
                            "  • LLM_AUTH_MODE：多为 bearer（Authorization: Bearer <key>）；\n"
                            "  • 若走 COMP/Azure APIM，请设置 LLM_BACKEND=comp，并使用 comp.azure-api.net + 部署名。\n"
                            f"当前请求：host={self.api_endpoint!r} POST path={self.request_path!r}"
                        ) from je
                    raise RuntimeError(
                        f"HTTP {status}: response is not JSON (first 500 chars): {text[:500]!r}"
                    ) from je

                if status and int(status) >= 400:
                    err_msg = json_data.get("error")
                    if isinstance(err_msg, dict):
                        err_msg = err_msg.get("message") or err_msg.get("type") or str(err_msg)
                    raise RuntimeError(
                        f"HTTP {status} from {self.api_endpoint}: {err_msg or text[:800]!r}"
                    )

                # 部分网关 HTTP 200 但 body 为 {"error": ...} 且无 choices
                if "choices" not in json_data and json_data.get("error") is not None:
                    err_obj = json_data["error"]
                    if isinstance(err_obj, dict):
                        err_detail = err_obj.get("message") or err_obj.get("type") or str(err_obj)
                    else:
                        err_detail = str(err_obj)
                    raise RuntimeError(
                        f"API error in body (no choices): {err_detail!r}; preview={text[:500]!r}"
                    )

                if "choices" not in json_data:
                    err_obj = json_data.get("error")
                    if isinstance(err_obj, dict):
                        err_detail = err_obj.get("message") or str(err_obj)
                    else:
                        err_detail = err_obj or "no 'choices' in response"
                    keys = list(json_data.keys())[:20]
                    raise RuntimeError(
                        f"API response missing 'choices' (OpenAI chat format). "
                        f"error={err_detail!r}; top-level keys={keys}; "
                        f"body_preview={text[:600]!r}"
                    )

                choice0 = json_data["choices"][0]
                msg = choice0.get("message") or {}
                content = msg.get("content")
                if content is None:
                    # 少数兼容：completion 风格
                    content = choice0.get("text")
                if content is None:
                    raise RuntimeError(
                        f"choices[0] has no message.content (keys: {list(choice0.keys())})"
                    )
                usage = json_data.get("usage")
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("API response content must be a nonempty string")
                trace["status"] = "success"
                _append_usage_event(
                    {
                        "request_id": request_id,
                        "event": "network_request",
                        "status": "success",
                        "attempt": attempt + 1,
                        "model": self.model,
                        "host": self.api_endpoint,
                        "request_path": self.request_path,
                        "temperature": self.temperature,
                        "http_status": int(status) if status is not None else None,
                        "latency_seconds": time.monotonic() - started,
                        "request_bytes": len(payload),
                        "response_bytes": len(raw),
                        "usage": usage if isinstance(usage, dict) else None,
                        "provider_model": json_data.get("model"),
                        "provider_request_id": json_data.get("id"),
                        "provider_cost": json_data.get("cost")
                        or json_data.get("estimated_cost"),
                    }
                )
                return content
            except Exception as e:
                safe_error = str(e).replace(self.api_key, "[REDACTED]") if self.api_key else str(e)
                trace.update(error_type=type(e).__name__, error=safe_error)
                _append_usage_event(
                    {
                        "request_id": request_id,
                        "event": "network_request",
                        "status": "error",
                        "attempt": attempt + 1,
                        "model": self.model,
                        "host": self.api_endpoint,
                        "request_path": self.request_path,
                        "temperature": self.temperature,
                        "latency_seconds": time.monotonic() - started,
                        "request_bytes": len(payload),
                        "error_type": type(e).__name__,
                    }
                )
                if self.debug_mode:
                    print(f"API call attempt {attempt + 1} failed: {safe_error}")
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed to get LLM response after {max_retries} attempts: {safe_error}") from None
            finally:
                trace["latency_seconds"] = time.monotonic() - started
                if conn is not None:
                    conn.close()
                _write_request_trace(trace, self.api_key)
        raise RuntimeError("Failed to get LLM response")


def _parse_extra_headers_from_env() -> dict:
    """从 LLM_EXTRA_HEADERS 读取 JSON，例如 {\"Accept\":\"application/json\"}"""
    raw = os.getenv("LLM_EXTRA_HEADERS", "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("LLM_EXTRA_HEADERS must be a JSON object")
        # 强制 key 为 str
        return {str(k): str(v) for k, v in data.items()}
    except json.JSONDecodeError as e:
        raise ValueError(
            "Invalid JSON in LLM_EXTRA_HEADERS. Example: "
            '{"Accept":"application/json"}'
        ) from e


def _parse_temperature(value: Optional[float]) -> float:
    raw = value if value is not None else os.getenv("LLM_TEMPERATURE", "0")
    try:
        temperature = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"LLM temperature must be numeric, got {raw!r}") from exc
    if not 0.0 <= temperature <= 2.0:
        raise ValueError(f"LLM temperature must be in [0, 2], got {temperature}")
    return temperature


def _cache_path(cache_payload: dict) -> Optional[Path]:
    cache_dir = os.getenv("LLM_CACHE_DIR", "").strip()
    mode = os.getenv("LLM_CACHE_MODE", "off").strip().lower()
    if not cache_dir or mode == "off":
        return None
    if mode not in {"readwrite", "readonly", "refresh"}:
        raise ValueError("LLM_CACHE_MODE must be one of: off, readwrite, readonly, refresh")
    canonical = json.dumps(
        cache_payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    return Path(cache_dir).expanduser().resolve() / f"{digest}.json"


def _read_cached_response(path: Optional[Path]) -> Optional[str]:
    if path is None or os.getenv("LLM_CACHE_MODE", "off").strip().lower() == "refresh":
        return None
    if not path.is_file():
        if os.getenv("LLM_CACHE_MODE", "off").strip().lower() == "readonly":
            raise FileNotFoundError(f"LLM cache miss in readonly mode: {path}")
        return None
    with path.open(encoding="utf-8") as f:
        record = json.load(f)
    response = record.get("response")
    if not isinstance(response, str):
        raise ValueError(f"Invalid LLM cache record: {path}")
    return response


def _write_cached_response(path: Optional[Path], response: str, metadata: dict) -> None:
    if path is None or os.getenv("LLM_CACHE_MODE", "off").strip().lower() == "readonly":
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f".{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump({"response": response, "metadata": metadata}, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def run_llm(prompt: str, 
            api_endpoint: Optional[str] = None,
            api_key: Optional[str] = None,
            model: Optional[str] = None,
            debug_mode: bool = False,
            max_retries: int = 3,
            auth_mode: Optional[str] = None,
            extra_headers: Optional[dict] = None,
            request_path: Optional[str] = None,
            temperature: Optional[float] = None) -> str:
    """
    调用LLM生成响应
    
    Args:
        prompt: 输入提示词
        api_endpoint: API端点（从环境变量或默认值获取）
        api_key: API密钥（从环境变量或默认值获取）
        model: 模型名称（从环境变量或默认值获取）
        debug_mode: 是否开启调试模式
        max_retries: 最大重试次数
    
    Returns:
        LLM响应字符串

    后端选择（环境变量 ``LLM_BACKEND``）：
        - 未设置或空：通用 OpenAI 兼容网关（Bearer / ``/v1/chat/completions``）。
        - ``jiekou`` / ``jiekou_ai``：与通用相同走 ``GeneralAPIInterface``；通常由
          ``--llm_backend 3`` 设置，默认路径 ``LLM_REQUEST_PATH=/openai/v1/chat/completions``。
        - ``comp`` / ``comp_openai`` / ``azure_apim`` / ``azure``：走项目内
          ``新的接口.comp_openai_client``（COMP Azure API Management，
          ``api-key`` 头 + ``/azure/openai/deployments/<部署名>/chat/completions``）。
          此时 ``api_endpoint`` 应为网关主机名（如 ``comp.azure-api.net``），
          ``model`` 为部署名（Deployment Name）；勿与通用网关的 ``LLM_REQUEST_PATH`` 混用。
    """
    # 从环境变量或参数获取配置（主机名须不含路径，避免误打到门户首页）
    api_endpoint = normalize_llm_api_host(
        api_endpoint or os.getenv("LLM_API_ENDPOINT", "api.bltcy.ai")
    )
    api_key = api_key or os.getenv("LLM_API_KEY", "")
    model = model or os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    auth_mode = auth_mode or os.getenv("LLM_AUTH_MODE", "bearer")
    merged_extra = _parse_extra_headers_from_env()
    if extra_headers:
        merged_extra = {**merged_extra, **extra_headers}
    req_path = request_path or os.getenv("LLM_REQUEST_PATH", "/v1/chat/completions")
    resolved_temperature = _parse_temperature(temperature)

    backend = (os.getenv("LLM_BACKEND") or "").strip().lower()
    if backend in ("comp", "comp_openai", "azure_apim", "azure"):
        if os.getenv("LLM_CACHE_MODE", "").strip().lower() == "readonly":
            raise ValueError("Legacy COMP backend has no offline replay support; refusing network access")
        comp = importlib.import_module("新的接口.comp_openai_client")
        rp = request_path if (request_path and str(request_path).strip()) else None
        return comp.run_llm(
            prompt,
            api_endpoint=api_endpoint,
            api_key=api_key,
            model=model,
            debug_mode=debug_mode,
            max_retries=max_retries,
            auth_mode=auth_mode,
            extra_headers=merged_extra if merged_extra else None,
            request_path=rp,
        )
    
    if not api_key:
        raise ValueError(
            "LLM API key not provided. Set LLM_API_KEY environment variable or pass api_key parameter.\n"
            "Example: export LLM_API_KEY='your-api-key'"
        )

    cache_payload = {
        "backend": backend or "openai_compatible",
        "host": api_endpoint,
        "model": model,
        "request_path": req_path,
        "temperature": resolved_temperature,
        "prompt": prompt,
    }
    global _CACHE_CALL_INDEX
    key_mode = os.getenv("LLM_CACHE_KEY_MODE", "call_sequence")
    if key_mode == "call_sequence":
        # Equal prompts at temperature > 0 are independent sampling operations.
        # A deterministic occurrence index allows replay without collapsing them.
        cache_payload.update(context=_CACHE_CONTEXT, call_index=_CACHE_CALL_INDEX)
        _CACHE_CALL_INDEX += 1
    elif key_mode != "prompt":
        raise ValueError("LLM_CACHE_KEY_MODE must be call_sequence or prompt")
    cache_path = _cache_path(cache_payload)
    cached = _read_cached_response(cache_path)
    if cached is not None:
        _append_usage_event(
            {
                "event": "cache_hit",
                "status": "success",
                "model": model,
                "host": api_endpoint,
                "request_path": req_path,
                "temperature": resolved_temperature,
                "response_bytes": len(cached.encode("utf-8")),
                "cache_key": cache_path.stem,
            }
        )
        return cached
    
    # 创建接口实例
    interface = GeneralAPIInterface(
        api_endpoint,
        api_key,
        model,
        debug_mode,
        auth_mode=auth_mode,
        extra_headers=merged_extra if merged_extra else None,
        request_path=req_path,
        temperature=resolved_temperature,
    )

    response = interface.get_response(prompt, max_retries=max_retries)
    _write_cached_response(
        cache_path,
        response,
        {
            **cache_payload,
            "cache_key_mode": key_mode,
        },
    )
    return response
