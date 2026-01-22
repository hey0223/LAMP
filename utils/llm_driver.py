import requests
import time
import asyncio
import aiohttp
import logging
import os

# Local vLLM API configuration
API_URL = "" 
MODEL_NAME = "" # TODO: Set your model name, e.g., "MiniCPM-V-2_6"
API_KEY = os.getenv("OPENROUTER_API_KEY", "") 

def get_headers():
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    return headers

HEADERS = get_headers()


def LLMCall(input_text, prompt="", temperature=0.7, max_tokens=1024, max_retries=3):
    """🔹 Synchronous call to local vLLM API"""
    if not API_URL:
        return "❌ Error: API_URL is not set in utils/llm_driver.py"
    messages = []
    if prompt:
        messages.append({"role": "system", "content": prompt})
    messages.append({"role": "user", "content": input_text})

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    for attempt in range(max_retries):
        try:
            headers = get_headers()
            response = requests.post(API_URL, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error: {e}, retrying in 5 seconds ({attempt + 1}/{max_retries})...")
                time.sleep(5)
            else:
                return f"LLM call failed: {str(e)}"


async def async_llm_call(input_text, prompt="", temperature=1.0, max_tokens=1024, max_retries=3):
    """🔹 Asynchronous call to local vLLM API (single request)"""
    messages = []
    if prompt:
        messages.append({"role": "system", "content": prompt})
    messages.append({"role": "user", "content": input_text})

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    for attempt in range(max_retries):
        try:
            headers = get_headers()
            async with aiohttp.ClientSession() as session:
                async with session.post(API_URL, json=payload, headers=headers) as resp:
                    resp.raise_for_status()
                    result = await resp.json()
                    return result["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < max_retries - 1:
                logging.warning(f"⚠️ Local server error, retrying in 5 seconds ({attempt + 1}/{max_retries})...")
                await asyncio.sleep(5)
            else:
                return f"❌ Request failed: {str(e)}"


async def LLMCallBatch(input_texts, prompt="", temperature=1.0, max_tokens=1024):
    """🚀 Batch asynchronous concurrent calls to local vLLM API"""
    tasks = [async_llm_call(text, prompt, temperature, max_tokens) for text in input_texts]
    return await asyncio.gather(*tasks)
