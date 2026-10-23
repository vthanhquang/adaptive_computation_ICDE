"""Clients for the official, OpenAI-compatible APIs of each model provider.

Every model id is routed to the endpoint of the company that serves it:

  deepseek-*  DeepSeek          https://api.deepseek.com/v1                         DEEPSEEK_API_KEY
  glm-*       Z.ai (Zhipu)      https://api.z.ai/api/paas/v4                        ZAI_API_KEY
  kimi-*      Moonshot AI       https://api.moonshot.ai/v1                          MOONSHOT_API_KEY
  qwen*       Alibaba Cloud     https://dashscope-intl.aliyuncs.com/compatible-mode/v1  DASHSCOPE_API_KEY
  hy*         Tencent Hunyuan   https://api.hunyuan.cloud.tencent.com/v1            HUNYUAN_API_KEY

A base URL can be overridden with <PROVIDER>_BASE_URL (e.g. DEEPSEEK_BASE_URL), for instance to
use a regional endpoint. Keys are read from the environment and never stored in the repository.
"""
from __future__ import annotations
import os, time

PROVIDERS = {
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "zai": ("https://api.z.ai/api/paas/v4", "ZAI_API_KEY"),
    "moonshot": ("https://api.moonshot.ai/v1", "MOONSHOT_API_KEY"),
    "dashscope": ("https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
    "hunyuan": ("https://api.hunyuan.cloud.tencent.com/v1", "HUNYUAN_API_KEY"),
}

# Model id prefix -> provider.
ROUTES = (("deepseek", "deepseek"), ("glm", "zai"), ("kimi", "moonshot"),
          ("qwen", "dashscope"), ("hy", "hunyuan"))

# Relative per-token price of each model (flash tier = 0.5), used for cost accounting.
RATE = {
    "hy3": 0.5, "deepseek-v4-flash": 0.5, "deepseek-v4.1-flash": 1.0,
    "deepseek-v4-pro": 1.0, "glm-5.3-flash": 1.0, "qwen3.8-max": 1.5, "kimi-k3": 2.0,
}


def provider_of(model):
    for prefix, name in ROUTES:
        if model.startswith(prefix):
            return name
    raise ValueError(f"no provider configured for model {model!r}")


def base_url(provider):
    return os.environ.get(f"{provider.upper()}_BASE_URL") or PROVIDERS[provider][0]


def api_key(provider):
    var = PROVIDERS[provider][1]
    key = os.environ.get(var)
    if key:
        return key
    try:  # Windows: variable set at user scope but not inherited by this process.
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            return winreg.QueryValueEx(k, var)[0]
    except (OSError, ImportError):
        raise RuntimeError(f"{var} not found")


_clients = {}


def client(provider):
    if provider not in _clients:
        from openai import OpenAI
        _clients[provider] = OpenAI(api_key=api_key(provider), base_url=base_url(provider),
                                    timeout=150, max_retries=1)
    return _clients[provider]


def chat(model, messages, temperature=0.0, max_tokens=4096):
    """Returns (text, usage dict, latency_ms)."""
    t0 = time.perf_counter()
    r = client(provider_of(model)).chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
    latency = (time.perf_counter() - t0) * 1000
    u = r.usage
    usage = {
        "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
    }
    return r.choices[0].message.content or "", usage, latency
