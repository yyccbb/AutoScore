import importlib.util
import json
import os
import sys
import re
from itertools import count
from pathlib import Path
from threading import Lock

import prompts as prompts
from utils_asro.progress import log_progress

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def _load_shared_call_llm():
    module_path = Path(__file__).resolve().parents[1] / "utils" / "llm_api.py"
    spec = importlib.util.spec_from_file_location("shared_llm_api", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.call_llm, module.call_llm_batch


_shared_call_llm, _shared_call_llm_batch = _load_shared_call_llm()


class GradeOptClient:
    def __init__(
        self,
        api_key=None,
        base_url="https://openrouter.ai/api/v1",
        grader_model="qwen/qwen-2.5-72b-instruct",
        reflector_model="deepseek/deepseek-r1",
        timeout=60.0,
        grader_temperature=0.0,
        reflector_temperature=0.7,
        grader_max_tokens=None,
        reflector_max_tokens=None,
        reflector_timeout=None,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.timeout = timeout
        self.reflector_timeout = reflector_timeout
        self.grader_model = grader_model
        self.model_reflector = reflector_model
        self.grader_temperature = grader_temperature
        self.reflector_temperature = reflector_temperature
        self.grader_max_tokens = grader_max_tokens
        self.reflector_max_tokens = reflector_max_tokens
        self._request_counter = count(1)
        self._request_lock = Lock()

    def _next_request_id(self):
        with self._request_lock:
            return next(self._request_counter)

    def call_llm(self, prompt, is_reflector=True):
        target_model = self.model_reflector if is_reflector else self.grader_model
        role = "reflector" if is_reflector else "grader"
        temperature = self.reflector_temperature if is_reflector else self.grader_temperature
        max_tokens = self.reflector_max_tokens if is_reflector else self.grader_max_tokens
        timeout = self.reflector_timeout if is_reflector and self.reflector_timeout is not None else self.timeout
        request_id = self._next_request_id()

        try:
            response = _shared_call_llm(
                model_name=target_model,
                system_prompt=None,
                user_prompt=prompt,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                extra_headers={
                    "HTTP-Referer": "https://ASRO-optimization.com",
                    "X-Title": "ASRO Evolution Engine",
                },
            )
        except Exception as exc:
            log_progress("llm", "request failed", request_id=request_id, role=role, model=target_model, error=exc)
            raise
        
        return response

    def _build_grader_prompt(self, essay_text, guideline):
        raw_gar = guideline.get("Gar", "")
        clean_gar = self._purify_gar(raw_gar)
        return prompts.GRADER_PROMPT_TEMPLATE.format(
            Gqs=guideline.get("Gqs", ""),
            Gsr=guideline.get("Gsr", ""),
            Gar=clean_gar,
            text=essay_text,
            max_score=guideline.get("max_score", 15),
            tier_count=guideline.get("tier_count", 5),
        )

    def get_ordinal_score(self, essay_text, guideline, true_score=None):
        user_prompt = self._build_grader_prompt(essay_text, guideline)
        max_score = guideline.get("max_score", 15)
        tier_count = guideline.get("tier_count", 5)
        attempt = 0

        while True:
            attempt += 1
            content = self.call_llm(user_prompt, is_reflector=False)
            try:
                score, grader_tags = self._parse_marker_response(
                    content,
                    max_score=max_score,
                    tier_count=tier_count,
                )
            except ValueError as exc:
                log_progress(
                    "grader_parse",
                    "grader response parse failed; retrying original prompt",
                    sample_id="unknown",
                    attempt=attempt,
                    model=self.grader_model,
                    error=self._abbreviate_error(exc),
                )
                continue
            return float(score), -1.0, -1.0, grader_tags

    def get_ordinal_score_batch(self, samples, guideline, max_workers=5, use_multithread=True):
        log_progress(
            "llm_batch",
            "grader batch started",
            samples=len(samples),
            workers=max_workers,
            multithread=use_multithread,
            model=self.grader_model,
        )
        request_items = [
            {
                "model_name": self.grader_model,
                "system_prompt": None,
                "user_prompt": self._build_grader_prompt(sample["text"], guideline),
                "api_key": self.api_key,
                "base_url": self.base_url,
                "temperature": self.grader_temperature,
                "max_tokens": self.grader_max_tokens,
                "timeout": self.timeout,
                "extra_headers": {
                    "HTTP-Referer": "https://ASRO-optimization.com",
                    "X-Title": "ASRO Evolution Engine",
                },
            }
            for sample in samples
        ]
        max_score = guideline.get("max_score", 15)
        tier_count = guideline.get("tier_count", 5)
        results = [None] * len(samples)
        attempts = [0] * len(samples)
        pending_indices = list(range(len(samples)))

        while pending_indices:
            pending_request_items = [request_items[idx] for idx in pending_indices]
            contents = _shared_call_llm_batch(
                pending_request_items,
                max_workers=max_workers,
                use_multithread=use_multithread,
            )
            if len(contents) != len(pending_indices):
                raise RuntimeError(
                    "Grader batch returned "
                    f"{len(contents)} response(s) for {len(pending_indices)} pending request(s)."
                )

            retry_indices = []
            for idx, content in zip(pending_indices, contents):
                attempts[idx] += 1
                try:
                    score, grader_tags = self._parse_marker_response(
                        content,
                        max_score=max_score,
                        tier_count=tier_count,
                    )
                except ValueError as exc:
                    sample = samples[idx]
                    sample_id = sample.get("id", "unknown") if isinstance(sample, dict) else "unknown"
                    log_progress(
                        "grader_parse",
                        "grader response parse failed; retrying original prompt",
                        sample_id=sample_id,
                        attempt=attempts[idx],
                        model=self.grader_model,
                        error=self._abbreviate_error(exc),
                    )
                    retry_indices.append(idx)
                    continue

                results[idx] = (float(score), -1.0, -1.0, grader_tags)

            pending_indices = retry_indices

        log_progress("llm_batch", "grader batch responses received", samples=len(results), model=self.grader_model)
        log_progress("llm_batch", "grader batch parsed", samples=len(results), model=self.grader_model)
        return results

    def _purify_gar(self, raw_gar):
        if not raw_gar:
            return ""
        if isinstance(raw_gar, dict):
            return "\n".join([f"### {k.upper()}\n{v}" for k, v in raw_gar.items()])
        text = str(raw_gar).strip()
        if "```" in text:
            text = re.sub(r"```[a-zA-Z]*\n?", "", text).replace("```", "").strip()
        try:
            if text.startswith("{"):
                data = json.loads(text)
                for key in ["full_refined_rubric", "Gar", "gar"]:
                    if key in data:
                        return str(data[key])
        except Exception:
            pass
        return text

    def _normalize_tag_name(self, tag_name):
        normalized = re.sub(r"[^A-Za-z0-9]+", "_", tag_name or "").strip("_").upper()
        if not normalized:
            raise ValueError(f"Invalid empty marker tag name: {tag_name!r}")
        return normalized

    def _abbreviate_error(self, exc, max_length=300):
        message = " ".join(str(exc).split())
        if len(message) <= max_length:
            return message
        return f"{message[:max_length - 3]}..."

    def _parse_marker_tags(self, content):
        if not content:
            raise ValueError("Empty grader response")
        text = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        marker_pattern = re.compile(
            r"\[\[\s*([^\[\]\r\n]+?)\s*\]\]\s*(?:[:：]\s*)?",
        )
        matches = list(marker_pattern.finditer(text))
        if not matches:
            raise ValueError("No marker tags found in grader response")

        tags = {}
        for idx, match in enumerate(matches):
            tag_name = self._normalize_tag_name(match.group(1))
            value_start = match.end()
            value_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            value = text[value_start:value_end].strip()

            if tag_name in ("SCORE", "TIER") and tag_name in tags:
                raise ValueError(f"Duplicate numeric grader tag: {tag_name}")
            if tag_name in tags:
                tags[tag_name] = f"{tags[tag_name]}\n\n{value}".strip()
            else:
                tags[tag_name] = value
        return tags

    def _parse_required_int_tag(self, tags, tag_name, min_value, max_value):
        if tag_name not in tags:
            raise ValueError(f"Missing required grader tag: {tag_name}")

        raw_value = str(tags[tag_name]).strip()
        if not re.fullmatch(r"\d+", raw_value):
            raise ValueError(f"{tag_name} must be an integer, got: {raw_value!r}")

        value = int(raw_value)
        if not min_value <= value <= max_value:
            raise ValueError(f"{tag_name} is out of range [{min_value}, {max_value}]: {value}")
        tags[tag_name] = value
        return value

    def _parse_marker_response(self, content, max_score=15, tier_count=5):
        max_score_int = int(float(max_score))
        tier_count_int = int(float(tier_count))
        if float(max_score_int) != float(max_score):
            raise ValueError(f"max_score must be an integer-compatible value, got: {max_score!r}")
        if float(tier_count_int) != float(tier_count):
            raise ValueError(f"tier_count must be an integer-compatible value, got: {tier_count!r}")

        tags = self._parse_marker_tags(content)
        score = self._parse_required_int_tag(tags, "SCORE", 0, max_score_int)
        tier = self._parse_required_int_tag(tags, "TIER", 0, tier_count_int)

        if score == 0 and tier != 0:
            raise ValueError(f"TIER must be 0 when SCORE is 0, got TIER={tier}")
        if score > 0 and tier == 0:
            raise ValueError(f"TIER can be 0 only when SCORE is 0, got SCORE={score}")

        return score, tags

    def request_diagnosis(self, diagnosis_prompt):
        return self.call_llm(diagnosis_prompt, is_reflector=True)
