import asyncio
import json
import logging
import os
import random
import re
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional
from uuid import uuid4

import aiohttp
import requests


# Qiniu OpenAI-compatible API configuration
API_URL = os.getenv("QINIU_API_URL", "")
MODEL_NAME = os.getenv("QINIU_MODEL_NAME", "")
API_KEY = os.getenv("QINIU_API_KEY", "")

# Runtime controls for throttling / retry
ASYNC_MAX_CONCURRENCY = max(1, int(os.getenv("LAMP_ASYNC_LLM_MAX_CONCURRENCY", "2")))
ASYNC_RPM = max(0, int(os.getenv("LAMP_ASYNC_LLM_RPM", "0")))
ASYNC_MIN_INTERVAL_SECONDS = float(
    os.getenv(
        "LAMP_ASYNC_LLM_MIN_INTERVAL_SECONDS",
        f"{(60.0 / ASYNC_RPM):.6f}" if ASYNC_RPM > 0 else "1.2",
    )
)
RETRY_BASE_SECONDS = max(0.5, float(os.getenv("LAMP_LLM_RETRY_BASE_SECONDS", "2")))
RETRY_CAP_SECONDS = max(RETRY_BASE_SECONDS, float(os.getenv("LAMP_LLM_RETRY_CAP_SECONDS", "60")))
REQUEST_TIMEOUT_SECONDS = max(10.0, float(os.getenv("LAMP_LLM_TIMEOUT_SECONDS", "120")))
DEFAULT_CALL_MAX_RETRIES = max(1, int(os.getenv("LAMP_LLM_CALL_MAX_RETRIES", "6")))
RATE_LIMIT_COOLDOWN_SECONDS = max(1.0, float(os.getenv("LAMP_LLM_429_COOLDOWN_SECONDS", "15")))
RATE_LIMIT_EXTRA_JITTER_SECONDS = max(0.0, float(os.getenv("LAMP_LLM_429_JITTER_SECONDS", "2")))


class AsyncRequestGate:
    """Global async gate to avoid bursty fan-out that triggers RPM 429s."""

    def __init__(self, max_concurrency: int, min_interval_seconds: float = 0.0):
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._interval_lock = asyncio.Lock()
        self._min_interval_seconds = max(0.0, min_interval_seconds)
        self._last_request_started_at = 0.0
        self._cooldown_until = 0.0

    async def impose_cooldown(self, seconds: float):
        seconds = max(0.0, float(seconds))
        if seconds <= 0:
            return
        async with self._interval_lock:
            self._cooldown_until = max(self._cooldown_until, time.monotonic() + seconds)

    async def wait_turn(self):
        async with self._interval_lock:
            now = time.monotonic()
            wait_s = 0.0
            if self._cooldown_until > now:
                wait_s = max(wait_s, self._cooldown_until - now)
            if self._min_interval_seconds > 0:
                wait_s = max(wait_s, self._min_interval_seconds - (now - self._last_request_started_at))
            if wait_s > 0:
                await asyncio.sleep(wait_s)
            self._last_request_started_at = time.monotonic()

    @asynccontextmanager
    async def slot(self):
        await self._semaphore.acquire()
        try:
            await self.wait_turn()
            yield
        finally:
            self._semaphore.release()


_ASYNC_REQUEST_GATE = AsyncRequestGate(
    max_concurrency=ASYNC_MAX_CONCURRENCY,
    min_interval_seconds=ASYNC_MIN_INTERVAL_SECONDS,
)


def get_headers():
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    return headers


def build_payload(
    input_text,
    prompt="",
    temperature=0.7,
    max_tokens=1024,
    tools=None,
    tool_choice=None,
    enable_thinking=False,
    thinking_budget_tokens=2048,
    extra_messages=None,
    extra_body=None,
):
    messages = []
    if prompt:
        messages.append({"role": "system", "content": prompt})

    if extra_messages:
        messages.extend(extra_messages)

    messages.append({"role": "user", "content": input_text})

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    if tools:
        payload["tools"] = tools

    if tool_choice is not None:
        payload["tool_choice"] = tool_choice

    if enable_thinking:
        payload["thinking"] = {
            "type": "enabled",
            "budget_tokens": thinking_budget_tokens,
        }

    if extra_body:
        payload.update(extra_body)

    return payload


def _extract_retry_after_seconds(headers: Optional[Dict[str, Any]], raw_text: str = "") -> Optional[float]:
    headers = headers or {}
    retry_after = None
    for key in ("Retry-After", "retry-after", "X-RateLimit-Reset-Requests", "x-ratelimit-reset-requests"):
        if key in headers:
            retry_after = headers.get(key)
            break

    if retry_after is not None:
        try:
            value = float(retry_after)
            if value >= 0:
                return value
        except Exception:
            pass

    if raw_text:
        m = re.search(r"retry(?:ing)?\s+in\s+([0-9]+(?:\.[0-9]+)?)\s+seconds", raw_text, flags=re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass

    return None


def _compute_backoff_seconds(attempt: int, retry_after_s: Optional[float] = None) -> float:
    exponential = min(RETRY_CAP_SECONDS, RETRY_BASE_SECONDS * (2 ** max(0, attempt - 1)))
    jitter = random.uniform(0.1, 1.0)
    if retry_after_s is not None:
        return max(0.0, min(RETRY_CAP_SECONDS, retry_after_s) + jitter)
    return min(RETRY_CAP_SECONDS, exponential + jitter)


def _compute_rate_limit_cooldown(retry_after_s: Optional[float] = None) -> float:
    base = retry_after_s if retry_after_s is not None else RATE_LIMIT_COOLDOWN_SECONDS
    return max(0.0, min(RETRY_CAP_SECONDS, float(base)) + random.uniform(0.0, RATE_LIMIT_EXTRA_JITTER_SECONDS))

def extract_content_from_result(result: dict) -> str:
    try:
        choice = result["choices"][0]
        message = choice.get("message", {}) or {}
        content = message.get("content", "")

        if content is None:
            return ""

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    else:
                        parts.append(json.dumps(item, ensure_ascii=False))
                else:
                    parts.append(str(item))
            return "\n".join(parts)

        return str(content)
    except Exception as e:
        raise ValueError(f"Invalid API response format: {e}")


def LLMCall(
    input_text,
    prompt="",
    temperature=0.7,
    max_tokens=1024,
    max_retries=DEFAULT_CALL_MAX_RETRIES,
    tools=None,
    tool_choice=None,
    enable_thinking=False,
    thinking_budget_tokens=2048,
    extra_messages=None,
    extra_body=None,
):
    """Synchronous call; returns string content."""
    call_id = uuid4().hex
    status_code = None

    if not API_URL:
        msg = "❌ Error: API_URL is not set"
        logging.error("[LLM_API_CONFIG_ERROR][sync] %s", msg)
        return '{"error":"API_URL is not set"}'

    payload = build_payload(
        input_text=input_text,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools,
        tool_choice=tool_choice,
        enable_thinking=enable_thinking,
        thinking_budget_tokens=thinking_budget_tokens,
        extra_messages=extra_messages,
        extra_body=extra_body,
    )

    for attempt in range(1, max_retries + 1):
        try:
            headers = get_headers()
            response = requests.post(API_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
            raw_text = response.text
            status_code = response.status_code

            if status_code >= 400:
                raise RuntimeError(f"HTTP {status_code} returned by API. Response preview: {raw_text[:500]}")

            result = response.json()
            content = extract_content_from_result(result)
            return content

        except Exception as e:
            is_last_attempt = attempt >= max_retries
            error_message = str(e)

            if not is_last_attempt:
                backoff_s = _compute_backoff_seconds(attempt)
                logging.warning(
                    "[LLM_API_RETRY][sync] type=%s attempt=%s/%s backoff=%.1fs detail=%s",
                    type(e).__name__,
                    attempt,
                    max_retries,
                    backoff_s,
                    error_message,
                )
                time.sleep(backoff_s)
            else:
                logging.error(
                    "[LLM_API_FINAL_ERROR][sync] type=%s attempts=%s/%s status=%s detail=%s",
                    type(e).__name__,
                    attempt,
                    max_retries,
                    status_code,
                    error_message,
                )
                return json.dumps({"error": f"LLM call failed: {error_message}"}, ensure_ascii=False)


async def async_llm_call(
    input_text,
    prompt="",
    temperature=0.7,
    max_tokens=1024,
    max_retries=DEFAULT_CALL_MAX_RETRIES,
    tools=None,
    tool_choice=None,
    enable_thinking=False,
    thinking_budget_tokens=2048,
    extra_messages=None,
    extra_body=None,
):
    """Async call; returns string content."""
    call_id = uuid4().hex

    if not API_URL:
        msg = "❌ Error: API_URL is not set"
        logging.error("[LLM_API_CONFIG_ERROR][async] %s", msg)
        return '{"error":"API_URL is not set"}'

    payload = build_payload(
        input_text=input_text,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools,
        tool_choice=tool_choice,
        enable_thinking=enable_thinking,
        thinking_budget_tokens=thinking_budget_tokens,
        extra_messages=extra_messages,
        extra_body=extra_body,
    )

    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)

    for attempt in range(1, max_retries + 1):
        status_code = None
        try:
            headers = get_headers()
            async with _ASYNC_REQUEST_GATE.slot():
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(API_URL, json=payload, headers=headers) as resp:
                        raw_text = await resp.text()
                        status_code = resp.status

                        if status_code >= 400:
                            retry_after_s = _extract_retry_after_seconds(dict(resp.headers), raw_text)
                            if status_code == 429:
                                await _ASYNC_REQUEST_GATE.impose_cooldown(_compute_rate_limit_cooldown(retry_after_s))
                            raise RuntimeError(f"HTTP {status_code} returned by API. Response preview: {raw_text[:500]}")

                        try:
                            result = json.loads(raw_text)
                        except json.JSONDecodeError as e:
                            raise ValueError(f"Response is not valid JSON: {e}; body={raw_text[:500]}")

                        content = extract_content_from_result(result)
                        return content

        except Exception as e:
            is_last_attempt = attempt >= max_retries
            error_message = str(e)
            retry_after_s = None
            m_status = re.search(r"HTTP\s+(\d+)", error_message)
            if m_status:
                try:
                    status_code = int(m_status.group(1))
                except Exception:
                    pass
            m_retry = re.search(r"retry(?:ing)?\s+in\s+([0-9]+(?:\.[0-9]+)?)\s+seconds", error_message, flags=re.IGNORECASE)
            if m_retry:
                try:
                    retry_after_s = float(m_retry.group(1))
                except Exception:
                    retry_after_s = None

            if status_code == 429:
                await _ASYNC_REQUEST_GATE.impose_cooldown(_compute_rate_limit_cooldown(retry_after_s))

            if not is_last_attempt:
                backoff_s = _compute_backoff_seconds(attempt, retry_after_s=retry_after_s)
                logging.warning(
                    "[LLM_API_RETRY][async] type=%s attempt=%s/%s status=%s backoff=%.1fs detail=%s",
                    type(e).__name__, attempt, max_retries, status_code, backoff_s, error_message
                )
                await asyncio.sleep(backoff_s)
            else:
                logging.error(
                    "[LLM_API_FINAL_ERROR][async] type=%s attempts=%s/%s status=%s detail=%s",
                    type(e).__name__, attempt, max_retries, status_code, error_message
                )
                return json.dumps({"error": f"Request failed: {error_message}"}, ensure_ascii=False)


async def LLMCallBatch(
    input_texts,
    prompt="",
    temperature=0.7,
    max_tokens=1024,
    tools=None,
    tool_choice=None,
    enable_thinking=False,
    thinking_budget_tokens=2048,
):
    tasks = []
    for text in input_texts:
        tasks.append(
            async_llm_call(
                text,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=tool_choice,
                enable_thinking=enable_thinking,
                thinking_budget_tokens=thinking_budget_tokens,
            )
        )
    return await asyncio.gather(*tasks)
