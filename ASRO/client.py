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
            Gkc=guideline.get("Gkc", ""),
            Gsr=guideline.get("Gsr", ""),
            Gar=clean_gar,
            text=essay_text,
            max_score=guideline.get("max_score", 15),
            tier_count=guideline.get("tier_count", 5),
        )

    def get_ordinal_score(self, essay_text, guideline, true_score=None):
        user_prompt = self._build_grader_prompt(essay_text, guideline)
        content = self.call_llm(user_prompt, is_reflector=False)
        score, reasoning = self._parse_marker_response(content)
        max_score = float(guideline.get("max_score", 15))
        score = min(max_score, max(0.0, float(score)))
        return score, -1.0, -1.0, reasoning

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
        contents = _shared_call_llm_batch(
            request_items,
            max_workers=max_workers,
            use_multithread=use_multithread,
        )
        log_progress("llm_batch", "grader batch responses received", samples=len(contents), model=self.grader_model)
        results = []
        max_score = float(guideline.get("max_score", 15))
        for content in contents:
            score, reasoning = self._parse_marker_response(content)
            score = min(max_score, max(0.0, float(score)))
            results.append((score, -1.0, -1.0, reasoning))
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

    def _parse_marker_response(self, content):
        if not content:
            return 0.0, "API Call Failed"

        text = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        score_pattern = r"(?:\[*SCORE\]*|分数)\s*[:：]\s*(\d+(?:\.\d+)?)"
        score_match = re.search(score_pattern, text, re.IGNORECASE)

        if score_match:
            score = float(score_match.group(1))
        else:
            first_num = re.search(r"(\d+(?:\.\d+)?)", text)
            score = float(first_num.group(1)) if first_num else 0.0

        tags = [r"\[*REASONING\]*", r"\[*CONTENT_EVIDENCE\]*", r"理由", r"依据"]
        combined_pattern = rf"(?:{'|'.join(tags)})\s*[:：]\s*(.*)"
        reason_match = re.search(combined_pattern, text, re.DOTALL | re.IGNORECASE)
        if reason_match:
            reasoning = reason_match.group(1).strip()
        else:
            parts = re.split(score_pattern, text, flags=re.IGNORECASE)
            reasoning = parts[-1].strip() if len(parts) > 1 else text[-100:]

        reasoning = re.sub(r"\[*TIER\]*\s*[:：]\s*\d+", "", reasoning, flags=re.IGNORECASE).strip()
        return score, reasoning

    def request_diagnosis(self, diagnosis_prompt):
        return self.call_llm(diagnosis_prompt, is_reflector=True)
