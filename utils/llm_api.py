from __future__ import annotations

import base64
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_DIR = str(Path(__file__).resolve().parents[1])
_shadowing_paths = []
for _path in ("", _PROJECT_DIR):
    if _path in sys.path:
        sys.path.remove(_path)
        _shadowing_paths.append(_path)
try:
    from openai import OpenAI
    from tqdm import tqdm
finally:
    for _path in reversed(_shadowing_paths):
        sys.path.insert(0, _path)

def _resolve_api_key(api_key: Optional[str]) -> str:
    resolved = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not resolved:
        raise ValueError("API key is required. Pass api_key or set OPENROUTER_API_KEY/OPENAI_API_KEY.")
    return resolved


def _image_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    if suffix in {".tif", ".tiff"}:
        return "image/tiff"
    return "image/jpeg"


def _encode_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _build_messages(system_prompt: Optional[str], user_prompt: str, images: Optional[List[Path]]):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if images:
        content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for image in images:
            image_path = Path(image)
            encoded = _encode_image(image_path)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{_image_mime(image_path)};base64,{encoded}"},
                }
            )
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": user_prompt})

    return messages


def call_llm(
    model_name: str,
    system_prompt: Optional[str],
    user_prompt: str,
    images: Optional[List[Path]] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: Optional[float] = None,
    response_format: Optional[dict] = None,
    extra_headers: Optional[dict] = None,
) -> str:
    client = OpenAI(api_key=_resolve_api_key(api_key), base_url=base_url)
    payload = {
        "model": model_name,
        "messages": _build_messages(system_prompt, user_prompt, images),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
        "response_format": response_format,
        "extra_headers": extra_headers,
    }
    payload = {k: v for k, v in payload.items() if v is not None}

    while True:
        try:
            response = client.chat.completions.create(**payload)
            result = response.choices[0].message.content
            if result is None or result.strip() == "":
                raise ValueError("No response from LLM")
            return result
        except Exception as e:
            print(e)            
            time.sleep(0.5)
            continue

def call_llm_batch(
    request_items: List[dict],
    max_workers: int = 5,
    use_multithread: bool = True,
):
    if not use_multithread or max_workers <= 1:
        results = []
        for item in tqdm(request_items, desc="LLM Batch", ncols=70):
            results.append(call_llm(**item))
        return results

    results = [None] * len(request_items)
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
        futures = {executor.submit(call_llm, **item): idx for idx, item in enumerate(request_items)}
        pbar = tqdm(total=len(futures), desc="LLM Batch", ncols=70)
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
            pbar.update(1)
        pbar.close()
    return results
