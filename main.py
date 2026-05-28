import copy
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys

_PROJECT_DIR = str(Path(__file__).resolve().parent)
_shadowing_paths = []
for _path in ("", _PROJECT_DIR):
    if _path in sys.path:
        sys.path.remove(_path)
        _shadowing_paths.append(_path)
try:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import cohen_kappa_score
    from tqdm import tqdm
    import yaml
finally:
    for _path in reversed(_shadowing_paths):
        sys.path.insert(0, _path)

from utils.env import load_env

load_env()

from utils.llm_api import call_llm
from utils.ocr import image_extensions, run_ocr_for_directory
import utils.prompts as prompts

ALLOWED_CHAT_KWARGS = {
    "temperature",
    "max_tokens",
    "timeout",
    "response_format",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "stop",
}


def _chat_kwargs(payload):
    return {k: v for k, v in payload.items() if k in ALLOWED_CHAT_KWARGS and v is not None}


class ScoringFailure(RuntimeError):
    def __init__(self, message, raw_response=None, stage=None):
        super().__init__(message)
        self.raw_response = raw_response
        self.stage = stage


def robust_extract_json(text: str):
    if not text:
        return {}
    
    text = re.sub(r'```json\n?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```[a-zA-Z]*\n?', '', text)
    text = re.sub(r'```\n?', '', text)
    
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and start_idx <= end_idx:
        try:
            return json.loads(text[start_idx : end_idx + 1])
        except json.JSONDecodeError as e:
            print(e)
    return {}


def extract_json_with_retry(call_fn, system_prompt, user_prompt, max_attempts=3, **payload):
    last_raw = None
    retry_system = system_prompt
    retry_user = user_prompt
    while True:
        last_raw = call_fn(retry_system, retry_user, **payload)
        parsed = robust_extract_json(last_raw)
        if parsed:
            return parsed, last_raw
        retry_system = (
            f"{system_prompt}\n\nYou must return one valid JSON object only. "
            "Do not include markdown fences, comments, or extra text."
        )
        retry_user = (
            f"{user_prompt}\n\nPrevious response was not valid JSON. "
            "Regenerate the answer as a single valid JSON object only."
        )

class AutoScoreClient:
    def __init__(
        self,
        api_key,
        base_url="https://openrouter.ai/api/v1",
        model_extraction="qwen/qwen3-235b-a22b-thinking-2507",
        model_scoring="qwen/qwen3-235b-a22b-thinking-2507",
        grader_model="qwen/qwen3-235b-a22b-thinking-2507",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model_extraction = model_extraction
        self.model_scoring = model_scoring
        self.grader = grader_model

    def call_llm(self, system_prompt, user_prompt, **payload):
        payload = _chat_kwargs(payload)
        return call_llm(
            model_name=self.grader,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            api_key=self.api_key,
            base_url=self.base_url,
            **payload,
        )

    def _call_llm(self, model, system_prompt, user_prompt, json_mode=False, max_tokens=1500, **kwargs):
        response_format = {"type": "json_object"} if json_mode else None
        payload = _chat_kwargs(kwargs)
        payload.setdefault("temperature", 0.1)
        payload.setdefault("max_tokens", max_tokens)
        if response_format is not None:
            payload["response_format"] = response_format
        return call_llm(
            model_name=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            api_key=self.api_key,
            base_url=self.base_url,
            **payload,
        )

    def autoscore_grade(self, task_rubric, essay_text, tier, required_points=None):
        if not required_points:
            raise ValueError(
                "AutoScore requires required_points derived from "
                "dataset.json subjective_question[qid].rubric."
            )
        
        extract_system = prompts.EXTRACTION_SYSTEM_PROMPT
        extract_user = prompts.EXTRACTION_USER_TEMPLATE.format(
            essay_text=essay_text,
            points=required_points
        )

        z_data, z_json_raw = extract_json_with_retry(
            lambda system, user, **payload: self._call_llm(
                self.model_extraction, system, user, max_tokens=4096, **payload
            ),
            extract_system,
            extract_user,
        )

        if not z_data:
            raise ScoringFailure(
                "AutoScore stage 1 JSON parse failed",
                raw_response=z_json_raw,
                stage="autoscore_extraction",
            )

        score_system = prompts.SCORING_SYSTEM_PROMPT
        score_user = prompts.SCORING_USER_TEMPLATE.format(
            task_rubric=task_rubric,
            evidence_z=json.dumps(z_data, ensure_ascii=False),
            essay_text=essay_text,
            tier=tier,
        )

        final_result, final_json_raw = extract_json_with_retry(
            lambda system, user, **payload: self._call_llm(
                self.model_scoring, system, user, max_tokens=2048, **payload
            ),
            score_system,
            score_user,
        )
        if not final_result:
            raise ScoringFailure(
                "AutoScore stage 2 JSON parse failed",
                raw_response=final_json_raw,
                stage="autoscore_scoring",
            )

        final_result["evidence_z"] = z_data  
        return final_result

def npx_converter(obj):
    if isinstance(obj, (np.int64, np.int32, np.int16)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def save_guideline(guideline, round_idx, qwk_score, is_best=False):
    save_dir = "optimized_guidelines"
    os.makedirs(save_dir, exist_ok=True)
    prefix = "BEST_" if is_best else ""
    filename = f"{prefix}round_{round_idx}_qwk_{qwk_score:.4f}"
    
    json_path = os.path.join(save_dir, f"{filename}.json")
    save_data = copy.deepcopy(guideline)
    save_data['round'] = int(round_idx)
    save_data['qwk'] = float(qwk_score)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=4, default=npx_converter)
    print(f"[OK] Saved guideline to {json_path}")

def save_error_logs(results, round_idx, top_n=10):
    log_dir = "error_analysis_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    worst_samples = sorted(results, key=lambda x: x['misconf'], reverse=True)[:top_n]
    
    file_path = os.path.join(log_dir, f"round_{round_idx}_bad_cases.md")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"# Round {round_idx} Error Analysis\n\n")
        f.write(f"> Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for i, s in enumerate(worst_samples):
            f.write(f"## Case {i+1} | Misconf: {s['misconf']:.2f}\n")
            f.write(f"- **ID:** `{s['id']}`\n") 
            f.write(f"- **True:** {s['true']} | **Pred:** {s['pred']}\n")
            f.write(f"### Essay:\n> {s['text']}\n\n")
            f.write(f"### Reasoning:\n```text\n{s['reasoning']}\n```\n")
            f.write(f"\n---\n")
            
    print(f"[OK] Bad cases saved to {file_path}")

def print_round_dashboard(round_idx, results, qwk):
    probs = [r.get('prob', 0.5) for r in results]
    sorted_samples = sorted(results, key=lambda x: x['misconf'], reverse=True)
    
    print(f"\n" + "="*70)
    print(f"[ROUND {round_idx} DASHBOARD]")
    print(f"[OK] QWK (Current): {qwk:.4f} | Avg Confidence: {np.mean(probs):.2f}")
    print(f"\nTop-5 Problematic Samples:")
    for s in sorted_samples[:5]:
        print(f"{s['true']:6.1f} | {s['pred']:6.1f} | {s['misconf']:10.2f}")
    print("="*70 + "\n")

def _score_to_tier(score, max_score=15.0, tier_count=5):
    if score <= 0:
        return 0
    step = float(max_score) / float(tier_count)
    return min(tier_count, max(1, int(np.ceil(float(score) / step))))


def _format_tier_ranges(max_score=15.0, tier_count=5):
    step = float(max_score) / float(tier_count)
    ranges = []
    for tier in range(1, int(tier_count) + 1):
        low = 0.0 if tier == 1 else (tier - 1) * step
        high = tier * step
        if tier == 1:
            ranges.append(f"Tier {tier}: 0 < score <= {high:.1f}")
        else:
            ranges.append(f"Tier {tier}: {low:.1f} < score <= {high:.1f}")
    return "\n".join(ranges)
    
def _parse_essay_stem(stem):
    parts = stem.split('_')
    if len(parts) >= 3 and parts[-2].isdigit() and len(parts[-2]) <= 3:
        q_idx = len(parts) - 2
    elif len(parts) >= 2 and parts[-1].isdigit() and len(parts[-1]) <= 3:
        q_idx = len(parts) - 1
    else:
        q_idx = None
    if q_idx is None:
        return None, None, None
    student_id = "_".join(parts[:q_idx])
    q_id = parts[q_idx]
    score = "_".join(parts[q_idx + 1:]) or None
    return student_id, q_id, score


def _rubric_for_question(q_id):
    raise ValueError(
        f"Missing rubric for question {q_id}. "
        "Rubrics must be loaded from dataset.json subjective_question[qid].rubric."
    )


def _required_points_from_rubric(q_id):
    raise ValueError(
        f"Missing required points for question {q_id}. "
        "Required points must be derived from dataset.json subjective_question[qid].rubric."
    )


def _format_dataset_task_context(q_id, qinfo):
    if not isinstance(qinfo, dict):
        return None

    sections = []
    fields = [
        ("Scoring Rubric", qinfo.get("rubric")),
    ]
    for title, value in fields:
        if value:
            if not isinstance(value, str):
                value = json.dumps(value, ensure_ascii=False, indent=2)
            sections.append(f"### {title} ({q_id})\n{value}")

    return "\n\n".join(sections) if sections else None


def _required_points_from_question_info(q_id, qinfo):
    if not isinstance(qinfo, dict):
        return None
    source_parts = []
    value = qinfo.get("rubric")
    if value:
        source_parts.append(str(value))
    if not source_parts:
        return None
    description = "\n\n".join(source_parts)
    return json.dumps(
        [{"id": f"question_{q_id}_requirements", "description": description}],
        ensure_ascii=False,
    )
    
def _target_ids(target_id):
    if not target_id:
        return None
    return {str(x).strip() for x in str(target_id).split(",") if str(x).strip()}


def _is_target_question(path, target_ids):
    if not target_ids:
        return True
    _, q_id, _ = _parse_essay_stem(path.stem)
    return q_id in target_ids
    
def _image_extensions():
    return image_extensions()


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def load_yaml_config(config_path, section):
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cfg = data.get(section)
    if cfg is None:
        raise KeyError(f"Config file missing `{section}` section")
    if not isinstance(cfg, dict):
        raise TypeError(f"`{section}` section must be a YAML mapping")
    return cfg
    
def _legacy_cleanup(raw_text):
    if not raw_text:
        return ""
    
    text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
    text = re.sub(r'^```[a-zA-Z]*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?```$', '', text, flags=re.MULTILINE)
    
    return text.strip()


def get_essay_tier(
    client,
    essay_text,
    guideline,
    guideline_system,
    max_score=15.0,
    tier_count=5,
    max_attempts=2,
    **telemetry,
):
    payload = {k: v for k, v in telemetry.items() if v is not None}
    if not essay_text or len(str(essay_text).strip()) < 10:
        return 0

    retry_guideline = guideline
    attempts = max(1, int(max_attempts))
    for attempt in range(attempts):
        try:
            raw_res = client.call_llm(guideline_system, retry_guideline, **payload)
            if "Error" in raw_res:
                raise RuntimeWarning("Upstream inference node jitter detected.")

            tier_match = re.search(r'\[*TIER\]*\s*[:：]\s*(\d+)', raw_res, re.I)
            if tier_match:
                tier = int(tier_match.group(1))
                if 1 <= tier <= int(tier_count):
                    return tier

            fallback_match = re.search(r'\b(\d+)\b', raw_res)
            if fallback_match:
                tier = int(fallback_match.group(1))
                if 1 <= tier <= int(tier_count):
                    return tier

            retry_guideline = (
                f"{guideline}\n\nThe previous answer did not contain a valid tier. "
                f"Return exactly one line: [[TIER]]: X, where X is an integer from 1 to {tier_count}."
            )
        except Exception:
            if attempt + 1 >= attempts:
                break

    word_count = len(str(essay_text).split())
    midpoint = max(1, int(np.ceil(float(tier_count) / 2.0)))
    if 80 <= word_count <= 100:
        return min(int(tier_count), midpoint + 1)
    if word_count >= 60:
        return midpoint
    return max(1, midpoint - 1)


def _failure_record(txt_path, dataset_name, student_id, q_id, stage, message, raw_response=None):
    return {
        "failed": 1,
        "dataset_name": dataset_name,
        "file": str(txt_path),
        "student_id": student_id,
        "q_id": q_id,
        "stage": stage,
        "error": str(message),
        "raw_response": raw_response or "",
    }

def _format_score_for_filename(score):
    """
    Format score for filename suffix.
    Examples:
      12.0 -> "12.0"
      9.5  -> "9.5"
      10   -> "10.0"
    """
    score = float(score)
    s = f"{score:.4f}".rstrip("0").rstrip(".")
    if "." not in s:
        s += ".0"
    return s


def append_truth_scores_to_filenames(
    input_dir,
    truth_lookup,
    dataset_name,
    target_ids=None,
    file_exts=(".txt",),
    strict=False,
):
    """
    Rename files by appending ground-truth score from dataset.json.

    Example:
      301010214_66.txt -> 301010214_66_9.5.txt

    Rename OCR text files into the score-annotated filename convention.
    """
    input_path = Path(input_dir)
    renamed = []
    skipped = []
    failed = []
    allowed_exts = {str(ext).lower() for ext in file_exts}

    for path in sorted(input_path.iterdir()):
        if not path.is_file():
            continue

        if path.suffix.lower() not in allowed_exts:
            continue

        if path.name.endswith("_cmp.jpg"):
            continue

        student_id, q_id, existing_score = _parse_essay_stem(path.stem)

        if not student_id or not q_id:
            skipped.append((path.name, "cannot_parse_filename"))
            continue

        if target_ids and q_id not in target_ids:
            skipped.append((path.name, "not_target_question"))
            continue

        # Already has score suffix, do not append again.
        if existing_score is not None:
            skipped.append((path.name, "already_has_score"))
            continue

        score = truth_lookup.get((dataset_name, student_id, q_id))
        if score is None:
            msg = f"missing_score: {dataset_name}/{student_id}/{q_id}"
            if strict:
                raise ValueError(msg)
            skipped.append((path.name, msg))
            continue

        score_str = _format_score_for_filename(score)
        new_path = path.with_name(f"{path.stem}_{score_str}{path.suffix}")

        if new_path.exists():
            msg = f"target_exists: {new_path.name}"
            skipped.append((path.name, msg))
            continue

        try:
            path.rename(new_path)
            renamed.append((path.name, new_path.name))
        except Exception as exc:
            if strict:
                raise
            failed.append((path.name, str(exc)))

    print(
        f"Score filename annotation: renamed={len(renamed)}, "
        f"skipped={len(skipped)}, failed={len(failed)}"
    )

    if failed:
        print("Filename annotation failures:")
        for name, err in failed[:10]:
            print(f"   - {name}: {err}")

    return {"renamed": renamed, "skipped": skipped, "failed": failed}


def _expected_scored_txt_path_for_image(in_path, image_path, truth_lookup, dataset_name):
    student_id, q_id, existing_score = _parse_essay_stem(image_path.stem)
    if not student_id or not q_id or existing_score is not None:
        return None

    score = truth_lookup.get((dataset_name, student_id, q_id))
    if score is None:
        return None

    score_str = _format_score_for_filename(score)
    return Path(in_path) / f"{student_id}_{q_id}_{score_str}.txt"

def load_dataset_metadata(json_path: Path):
    if not json_path.exists():
        return {}, {}, {}
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    score_lookup = {}
    task_lookup = {}
    points_lookup = {}
    for dataset_name, dataset_content in data.items():
        questions = dataset_content.get("subjective_question", {})
        if isinstance(questions, dict):
            for q_id, qinfo in questions.items():
                q_id = str(q_id)
                task_context = _format_dataset_task_context(str(q_id), qinfo)
                if task_context:
                    task_lookup[(dataset_name, q_id)] = task_context
                required_points = _required_points_from_question_info(q_id, qinfo)
                if required_points:
                    points_lookup[(dataset_name, q_id)] = required_points

        if "student_answer" in dataset_content:
            for student in dataset_content["student_answer"]:
                student_id = str(student.get("id"))
                grading = student.get("grading", {})
                for q_id, score in grading.items():
                    q_id = str(q_id)
                    try:
                        score_lookup[(dataset_name, student_id, q_id)] = float(score)
                    except (ValueError, TypeError):
                        pass
    return score_lookup, task_lookup, points_lookup


def load_ground_truth(json_path: Path):
    score_lookup, _, _ = load_dataset_metadata(json_path)
    return score_lookup

def load_optimized_rules(rule_file: str):
    if not rule_file or not os.path.exists(rule_file):
        return ""
    try:
        with open(rule_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            rules = data.get("optimized_rule", data.get("rules", ""))
            if isinstance(rules, (dict, list)):
                rules = json.dumps(rules, ensure_ascii=False, indent=2)
            return rules
    except Exception:
        return ""

def clean_json_response(response_text):
    try:
        json_str = re.search(r'\{.*\}', response_text, re.DOTALL).group()
        return json.loads(json_str)
    except:
        return None

def process_single_essay(
    txt_path,
    truth_lookup,
    task_lookup,
    points_lookup,
    optimized_rules,
    client,
    res_dir,
    dataset_name,
    temp,
    max_tokens,
    timeout,
    mode="autoscore",
    max_score=15.0,
    tier_count=5,
    tier_retries=2,
    strict=False,
):
    stem = txt_path.stem
    student_id, q_id, bias = _parse_essay_stem(stem)
    if not student_id or not q_id:
        message = f"Cannot parse student id or question id from filename: {txt_path.name}"
        if strict:
            raise ValueError(message)
        return _failure_record(txt_path, dataset_name, student_id, q_id, "filename_parse", message)
    
    try: 
        static = float(bias)
    except (ValueError, TypeError): 
        static = truth_lookup.get((dataset_name, student_id, q_id))
        
    if static is None:
        message = f"Missing true score for {dataset_name}/{student_id}/{q_id}"
        if strict:
            raise ValueError(message)
        return _failure_record(txt_path, dataset_name, student_id, q_id, "missing_score", message)

    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            student_text = f.read().strip()

        current_rubric = task_lookup.get((dataset_name, q_id)) or _rubric_for_question(q_id)
        if mode == "baseline":
            tier = None
        else:
            tier_prompt = prompts.TIER_CLASSIFIER_PROMPT.format(
                text=student_text,
                rubric=current_rubric,
                file_name=txt_path.name,
                max_score=max_score,
                tier_count=tier_count,
                tier_ranges=_format_tier_ranges(max_score, tier_count),
            )

            tier = get_essay_tier(
                client=client,
                essay_text=student_text,
                guideline=tier_prompt,
                guideline_system=prompts.TIER_SYSTEM_PROMPT.format(
                    tier_count=tier_count, 
                    tier_count_2=int(tier_count*0.4), 
                    tier_count_3=int(tier_count*0.6), 
                    tier_count_4=int(tier_count*0.8)
                ),
                max_score=max_score,
                tier_count=tier_count,
                max_attempts=tier_retries,
                temperature=temp,
                max_tokens=max_tokens,
                timeout=timeout,
                response_format={"type": "text"},
            )

        dynamic_rules = ""
        if optimized_rules:
            try: dynamic_rules = optimized_rules.format(tier=tier)
            except: dynamic_rules = optimized_rules

        if mode == "ASRO":
            scoring_prompt = prompts.PURE_TEMPLATE.format(
                task_rubric=current_rubric, 
                extra_rules=dynamic_rules, 
                essay_text=student_text, 
                tier=tier
            )
            final_res, score_raw = extract_json_with_retry(
                client.call_llm,
                prompts.SCORING_SYSTEM_PROMPT,
                scoring_prompt,
            )
            if not final_res:
                raise ScoringFailure("ASRO scoring JSON parse failed", raw_response=score_raw, stage="asro_scoring")
            ai_score = float(final_res.get("total_score", 0))

        elif mode == "baseline":
            scoring_prompt = prompts.BASELINE_PROMPT.format(
                rubric=current_rubric,
                essay_text=student_text,
            )
            final_res, score_raw = extract_json_with_retry(
                client.call_llm,
                prompts.SCORING_SYSTEM_PROMPT,
                scoring_prompt,
            )
            if not final_res:
                raise ScoringFailure("Baseline scoring JSON parse failed", raw_response=score_raw, stage="baseline_scoring")
            ai_score = float(final_res.get("total_score", 0))

        elif mode == "autoscore":
            current_required_points = points_lookup.get((dataset_name, q_id)) or _required_points_from_rubric(q_id)
            final_res = client.autoscore_grade(
                task_rubric=current_rubric, 
                essay_text=student_text, 
                tier=tier,
                required_points=current_required_points,
            )
            ai_score = float(final_res.get("total_score", 0))
        else:
            raise ValueError(f"Unknown run mode: {mode}")

        # =======================================================
        # Unified output metrics.
        # =======================================================
        abs_error = abs(ai_score - static)

        return {
            "student_id": student_id,
            "q_id": q_id,
            "true_score": static,
            "ai_score": ai_score,
            "diff": ai_score - static,
            "abs_error": abs_error,
            "is_anomaly": 1 if abs_error >= 3.0 else 0,
            "tier": tier,
            "stem": stem
        }
    except ScoringFailure as e:
        if strict:
            raise
        return _failure_record(txt_path, dataset_name, student_id, q_id, e.stage or "scoring", e, e.raw_response)
    except Exception as e:
        if strict:
            raise
        return _failure_record(txt_path, dataset_name, student_id, q_id, "processing", e)

def run_autoscore_pipeline(
    input_dir, out_dir, json_path, dataset_name, target_id, rules_file=None, num=None,
    workers=5, temp=0.1, max_tokens=1500, timeout=45, api_key=None,
    enable_ocr=True, force_ocr=False, ocr_only=False, debug=True, mode="autoscore",
    base_url="https://openrouter.ai/api/v1",
    ocr_model="qwen/qwen-2-vl-72b-instruct",
    model_extraction="qwen/qwen3-235b-a22b-thinking-2507",
    model_scoring="qwen/qwen3-235b-a22b-thinking-2507",
    grader_model="qwen/qwen3-235b-a22b-thinking-2507",
    max_score=15.0,
    tier_count=5,
    tier_retries=2,
    strict=False,
):
    in_path = Path(input_dir)
    res_dir = Path(out_dir)
    res_dir.mkdir(parents=True, exist_ok=True)

    final_api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    client = AutoScoreClient(
        api_key=final_api_key,
        base_url=base_url,
        model_extraction=model_extraction,
        model_scoring=model_scoring,
        grader_model=grader_model,
    )

    target_q = _target_ids(target_id)
    truth_lookup, task_lookup, points_lookup = load_dataset_metadata(Path(json_path))
    optimized_rules = load_optimized_rules(rules_file)

    if truth_lookup:
        append_truth_scores_to_filenames(
            in_path,
            truth_lookup,
            dataset_name,
            target_ids=target_q,
            strict=strict,
        )

    all_paths = [p for p in in_path.glob("*.txt") if _is_target_question(p, target_q)]

    if enable_ocr and in_path.exists():
        def expected_txt_resolver(image_path):
            image_path = Path(image_path)
            if not _is_target_question(image_path, target_q):
                return "skip"

            txt_out = image_path.with_suffix(".txt")
            scored_txt_out = _expected_scored_txt_path_for_image(
                in_path, image_path, truth_lookup, dataset_name
            )
            if scored_txt_out:
                return scored_txt_out
            return txt_out
        
        ocr_result = run_ocr_for_directory(
            input_dir=in_path,
            model_name=ocr_model,
            api_key=final_api_key,
            base_url=base_url,
            force_ocr=force_ocr,
            debug=debug,
            workers=workers,
            use_multithread=True,
            skip_existing=True,
            expected_txt_resolver=expected_txt_resolver,
            limit=num,
        )
        if ocr_result["failed"]:
            print(f"OCR failed for {len(ocr_result['failed'])} image(s); see error_report_ocr.csv.")

        all_paths = [p for p in in_path.glob("*.txt") if _is_target_question(p, target_q)]

        if not all_paths:
            print("No matching txt files are available after OCR; stopping pipeline.")
            return None
    elif not enable_ocr:
        all_paths = [p for p in in_path.glob("*.txt") if _is_target_question(p, target_q)]
        if not all_paths:
            raise RuntimeError(
                "No matching txt files found. Set enable_ocr=true or run OCR before scoring."
            )

    if truth_lookup:
        append_truth_scores_to_filenames(
            in_path,
            truth_lookup,
            dataset_name,
            target_ids=target_q,
            strict=strict,
        )
        all_paths = [p for p in in_path.glob("*.txt") if _is_target_question(p, target_q)]

    if ocr_only:
        print("OCR-only mode complete; skipping scoring.")
        return None
    
    if num: all_paths = all_paths[:num]
    analysis_data = []
    error_data = []

    print(f"[INFO] Scoring {len(all_paths)} files with workers={workers}, mode={mode}")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                process_single_essay, 
                p, truth_lookup, task_lookup, points_lookup, optimized_rules, client, res_dir,
                dataset_name, temp, max_tokens, timeout, mode, max_score, tier_count, tier_retries, strict
            ) for p in all_paths
        ]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Scoring", ncols=70):
            res = future.result()
            if not res:
                continue
            if res.get("failed"):
                error_data.append(res)
            else:
                analysis_data.append(res)

    if error_data:
        error_df = pd.DataFrame(error_data)
        error_path = res_dir / f"error_report_{dataset_name}.csv"
        error_df.to_csv(error_path, index=False, encoding='utf-8-sig')
        print(f"\nFailed samples: {len(error_data)} | error report: {error_path}")

    if analysis_data:
        df = pd.DataFrame(analysis_data)
        csv_path = res_dir / f"results_{dataset_name}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        y_true = [int(round(t*2)) for t in df['true_score']]
        y_pred = [int(round(p*2)) for p in df['ai_score']]
        qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        metrics_str = f"QWK: {qwk:.4f} | MAE: {df['abs_error'].mean():.4f} | failures: {len(error_data)}"
        with open(csv_path, "a", encoding="utf-8-sig") as f:
            f.write("\n" + metrics_str + "\n")
        print(f"\nPipeline complete. {metrics_str}")
        
        return df
        
    return None

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="[AES] OCR/AutoScore/ASRO inference pipeline")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument("--section", type=str, default="pipeline", help="YAML 配置段名称，默认 pipeline")
    parser.add_argument("--num", type=int, default=None, help="Limit OCR/scoring to this many samples, overriding config num.")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config, args.section)
    num = args.num if args.num is not None else cfg.get("num")

    run_autoscore_pipeline(
        input_dir=cfg["input_dir"],
        out_dir=cfg["out_dir"],
        json_path=cfg.get("json_path", "./data/dataset.json"),
        rules_file=cfg.get("rules_file"),
        target_id=cfg.get("target_id"),
        dataset_name=cfg.get("dataset_name", "English_Grade12_003"),
        num=num,
        workers=cfg.get("workers", 5),
        temp=cfg.get("temp", 0.1),
        max_tokens=cfg.get("max_tokens", 1500),
        timeout=cfg.get("timeout", 45),
        api_key=cfg.get("api_key"),
        enable_ocr=cfg.get("enable_ocr", cfg.get("ocr", True)),
        force_ocr=cfg.get("force_ocr", False),
        ocr_only=cfg.get("ocr_only", False),
        debug=cfg.get("debug", False),
        mode=cfg.get("mode", "ASRO"),
        base_url=cfg.get("base_url", "https://openrouter.ai/api/v1"),
        ocr_model=cfg.get("ocr_model", "qwen/qwen-2-vl-72b-instruct"),
        model_extraction=cfg.get("model_extraction", "qwen/qwen3-235b-a22b-thinking-2507"),
        model_scoring=cfg.get("model_scoring", "qwen/qwen3-235b-a22b-thinking-2507"),
        grader_model=cfg.get("grader_model", "qwen/qwen3-235b-a22b-thinking-2507"),
        max_score=cfg.get("max_score", 15.0),
        tier_count=cfg.get("tier_count", 5),
        tier_retries=cfg.get("tier_retries", 2),
        strict=cfg.get("strict", False),
    )
