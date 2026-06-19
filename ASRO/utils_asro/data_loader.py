import os
import json
import random
import re
from pathlib import Path
from collections import defaultdict

from utils_asro.progress import log_progress

class GradeOptDataLoader:
    def __init__(
        self,
        json_path,
        dataset_name="English_Grade12_004",
        max_score=15.0,
        sample_filter_enabled=False,
        sample_filter_model=None,
        api_key=None,
        base_url="https://openrouter.ai/api/v1",
        timeout=60.0,
        max_workers=5,
        task_context=None,
    ):
        self.json_path = Path(json_path)
        self.dataset_name = dataset_name
        self.max_score = float(max_score)
        self.sample_filter_enabled = bool(sample_filter_enabled)
        self.sample_filter_model = sample_filter_model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_workers = max(1, int(max_workers))
        self.task_context = task_context or {}
        self.score_lookup = self._load_scores()

    def _load_scores(self):
        """解析 dataset.json 建立分数映射"""
        if not self.json_path.exists():
            raise FileNotFoundError(f"找不到分数文件: {self.json_path}")
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        lookup = {}
        content = data.get(self.dataset_name, {})
        for student in content.get("student_answer", []):
            s_id = str(student.get("id"))
            grading = student.get("grading", {})
            for q_id, score in grading.items():
                lookup[(s_id, str(q_id))] = float(score)
        return lookup

    def _parse_txt_stem(self, stem):
        parts = stem.split("_")
        if len(parts) >= 3 and parts[-2].isdigit() and len(parts[-2]) <= 3:
            q_idx = len(parts) - 2
        elif len(parts) >= 2 and parts[-1].isdigit() and len(parts[-1]) <= 3:
            q_idx = len(parts) - 1
        else:
            return None, None
        return "_".join(parts[:q_idx]), parts[q_idx]

    def _get_tier(self, score, q_id="66"):
        """Use max_score-based 50% and 75% split points."""
        if score <= self.max_score * 0.5:
            return "Low"
        if score <= self.max_score * 0.75:
            return "Mid"
        return "High"

    def load_samples(self, all_txt_dir, q_id="66"):
        """Load usable OCR samples into a flat list before any split logic."""
        all_txt_dir = Path(all_txt_dir)
        q_id = str(q_id)
        samples = []

        # 兼容你的文件名格式: {id}_{q_id}_{score}.txt
        txt_files = list(all_txt_dir.glob(f"*_{q_id}_*.txt"))
        if not txt_files:
            # 兼容旧格式: {id}_{q_id}.txt
            txt_files = list(all_txt_dir.glob(f"*_{q_id}.txt"))

        for tf in txt_files:
            student_id, parsed_qid = self._parse_txt_stem(tf.stem)
            if not student_id or parsed_qid != q_id:
                print(f"Skipping unparsable OCR file: {tf.name}")
                continue
            true_score = self.score_lookup.get((student_id, q_id))

            if true_score is not None:
                tier = self._get_tier(true_score, q_id)
                with open(tf, "r", encoding="utf-8") as f:
                    content = f.read().strip()

                    if "[OCR Error]" in content or len(content) < 10:
                        print(f"⏭️ Skipping invalid OCR file: {tf.name}")
                        continue

                    samples.append({
                        "id": student_id,
                        "text": content,
                        "true_score": true_score,
                        "tier": tier
                    })

        return samples

    def _strip_json_fences(self, text):
        text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
        fenced = re.search(r"```json\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            return fenced.group(1).strip()
        text = re.sub(r"```[a-zA-Z]*\s*", "", text)
        return text.replace("```", "").strip()

    def _first_json_object(self, text):
        start = text.find("{")
        if start < 0:
            raise ValueError("No JSON object start found")

        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            char = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]

        raise ValueError("No complete JSON object found")

    def _parse_sample_filter_response(self, raw_response):
        json_text = self._first_json_object(self._strip_json_fences(raw_response))
        parsed = json.loads(json_text)
        if not isinstance(parsed, dict):
            raise ValueError("Sample filter response must be a JSON object")
        return {
            "blank": bool(parsed.get("blank", False)),
            "irrelevant_high_score": bool(parsed.get("irrelevant_high_score", False)),
            "reason": parsed.get("reason"),
        }

    def filter_samples(self, samples):
        """Use an opt-in LLM scan to remove blank or irrelevant high-score samples."""
        samples = list(samples)
        if not samples:
            return samples
        if not self.sample_filter_model:
            raise ValueError("sample_filter_model is required when sample filtering is enabled.")

        try:
            from utils.llm_api import call_llm_batch
            from utils.prompts import SAMPLE_FILTER_SYSTEM_PROMPT, SAMPLE_FILTER_USER_TEMPLATE
        except Exception as exc:
            log_progress("sample_filter", "LLM sample filter unavailable; keeping all samples", error=exc)
            return samples

        question = self.task_context.get("Gqs", "")
        rubric = self.task_context.get("Gsr", "")
        high_score_cutoff = self.max_score * 0.5
        request_items = []
        for sample in samples:
            user_prompt = SAMPLE_FILTER_USER_TEMPLATE.format(
                question=question,
                rubric=rubric,
                essay_text=sample.get("text", ""),
                sample_id=sample.get("id", "unknown"),
                true_score=sample.get("true_score"),
                max_score=self.max_score,
                high_score_cutoff=high_score_cutoff,
            )
            request_items.append({
                "model_name": self.sample_filter_model,
                "system_prompt": SAMPLE_FILTER_SYSTEM_PROMPT,
                "user_prompt": user_prompt,
                "api_key": self.api_key,
                "base_url": self.base_url,
                "temperature": 0.0,
                "timeout": self.timeout,
                "extra_headers": {
                    "HTTP-Referer": "https://ASRO-optimization.com",
                    "X-Title": "ASRO Sample Filter",
                },
            })

        log_progress("sample_filter", "LLM sample filter started", samples=len(samples), model=self.sample_filter_model)
        try:
            responses = call_llm_batch(request_items, max_workers=self.max_workers, use_multithread=self.max_workers > 1)
        except Exception as exc:
            log_progress("sample_filter", "LLM sample filter failed; keeping all samples", error=exc)
            return samples
        if len(responses) != len(samples):
            log_progress(
                "sample_filter",
                "LLM sample filter returned unexpected response count; keeping all samples",
                expected=len(samples),
                actual=len(responses),
            )
            return samples

        kept_samples = []
        dropped_blank = 0
        dropped_irrelevant = 0
        parse_failures = 0
        for sample, raw_response in zip(samples, responses):
            try:
                decision = self._parse_sample_filter_response(raw_response)
            except Exception as exc:
                parse_failures += 1
                log_progress(
                    "sample_filter",
                    "sample filter JSON parse failed; keeping sample",
                    sample_id=sample.get("id"),
                    error=exc,
                )
                kept_samples.append(sample)
                continue

            blank = bool(decision["blank"])
            irrelevant_high_score = bool(decision["irrelevant_high_score"])
            if blank and irrelevant_high_score:
                irrelevant_high_score = False

            try:
                true_score = float(sample.get("true_score", 0))
            except (TypeError, ValueError):
                true_score = 0.0
            if true_score <= high_score_cutoff:
                irrelevant_high_score = False

            if blank:
                dropped_blank += 1
                log_progress("sample_filter", "dropping blank sample", sample_id=sample.get("id"), reason=decision.get("reason"))
            elif irrelevant_high_score:
                dropped_irrelevant += 1
                log_progress(
                    "sample_filter",
                    "dropping irrelevant high-score sample",
                    sample_id=sample.get("id"),
                    score=true_score,
                    reason=decision.get("reason"),
                )
            else:
                kept_samples.append(sample)

        log_progress(
            "sample_filter",
            "LLM sample filter finished",
            input=len(samples),
            kept=len(kept_samples),
            dropped_blank=dropped_blank,
            dropped_irrelevant_high_score=dropped_irrelevant,
            parse_failures=parse_failures,
        )
        return kept_samples

    def downsample_samples(self, samples, sample_ratio=None):
        """Apply optional tier-balanced debug downsampling before splitting."""
        samples = list(samples)
        if sample_ratio is None:
            return samples

        sample_ratio = float(sample_ratio)
        if not 0 < sample_ratio <= 1:
            raise ValueError(f"sample_ratio must be in (0, 1] when set; got {sample_ratio}")
        if sample_ratio >= 1:
            return samples

        tier_buckets = self._bucket_samples_by_tier(samples)
        original_total = sum(len(tier_buckets[tier]) for tier in ["Low", "Mid", "High"])
        for tier in ["Low", "Mid", "High"]:
            tier_samples = tier_buckets[tier]
            if not tier_samples:
                continue
            random.shuffle(tier_samples)
            keep_count = max(1, int(round(len(tier_samples) * sample_ratio)))
            tier_buckets[tier] = tier_samples[:keep_count]

        filtered_samples = []
        for tier in ["Low", "Mid", "High"]:
            filtered_samples.extend(tier_buckets[tier])

        downsampled_total = len(filtered_samples)
        print(
            f"🐞 Debug sample ratio active: ratio={sample_ratio:.4f}, "
            f"usable={original_total}, downsampled={downsampled_total}"
        )
        return filtered_samples

    def _bucket_samples_by_tier(self, samples):
        """Group flat samples by score tier."""
        tier_buckets = defaultdict(list)
        for sample in samples:
            tier_buckets[sample.get("tier")].append(sample)
        return tier_buckets

    def split_balanced_samples(self, samples, train_size=None, val_size=None, val_ratio=0.25):
        """
        分层抽取平衡数据集。
        train_size 和 val_size 建议设为 3 的倍数，以便各档位平分。
        """
        tier_buckets = self._bucket_samples_by_tier(samples)
        D_train, D_val = [], []

        print(f"📊 数据池分布: Low={len(tier_buckets['Low'])}, Mid={len(tier_buckets['Mid'])}, High={len(tier_buckets['High'])}")

        for tier in ["Low", "Mid", "High"]:
            samples = tier_buckets[tier]
            random.shuffle(samples) # 随机打乱当前档位

            if train_size is None or val_size is None:
                v_per_tier = max(1, int(round(len(samples) * val_ratio))) if len(samples) > 1 else 0
                t_per_tier = max(0, len(samples) - v_per_tier)
            else:
                t_per_tier = train_size // 3
                v_per_tier = val_size // 3
                required = t_per_tier + v_per_tier
                if len(samples) < required:
                    print(f"⚠️ 警告: {tier} 档位样本不足 (需要 {required}, 实际 {len(samples)})，将按实际数量切分。")
                    v_per_tier = max(1, int(round(len(samples) * val_ratio))) if len(samples) > 1 else 0
                    t_per_tier = max(0, len(samples) - v_per_tier)

            D_train.extend(samples[:t_per_tier])
            D_val.extend(samples[t_per_tier : t_per_tier + v_per_tier])

        # 3. 最后打乱输出顺序，防止规律性影响
        random.shuffle(D_train)
        random.shuffle(D_val)

        print(f"✅ 平衡数据集加载完成: Train={len(D_train)} 篇, Val={len(D_val)} 篇")
        if not D_val:
            if len(D_train) > 1:
                D_val.append(D_train.pop())
                print("⚠️ D_val was empty; moved one training sample into validation.")
            else:
                raise ValueError("D_val 为空，无法进行 ASRO 验证。请增加数据量或调高 val_ratio。")
        return D_train, D_val

    def get_balanced_splits(self, all_txt_dir, q_id="66", train_size=None, val_size=None, val_ratio=0.25, sample_ratio=None):
        """
        Backward-compatible pipeline wrapper for loading, filtering, and splitting.
        """
        samples = self.load_samples(all_txt_dir, q_id=q_id)
        if self.sample_filter_enabled:
            samples = self.filter_samples(samples)
        samples = self.downsample_samples(samples, sample_ratio=sample_ratio)
        return self.split_balanced_samples(
            samples,
            train_size=train_size,
            val_size=val_size,
            val_ratio=val_ratio,
        )
