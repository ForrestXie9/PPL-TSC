"""Safe defaults for a generic OpenAI-compatible endpoint.

Credentials are intentionally not stored in source code. Set ``LLM_API_KEY``
in the environment or use the public ``run.py`` entry point.
"""

GENERAL_API_ENDPOINT = "127.0.0.1:8000"
GENERAL_API_KEY = ""
GENERAL_MODEL = "qwen2.5-32b"
GENERAL_REQUEST_PATH = "/v1/chat/completions"
GENERAL_AUTH_MODE = "bearer"
GENERAL_TEMPERATURE = "0"
