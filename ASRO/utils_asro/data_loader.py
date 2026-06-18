import os
import json
import random
from pathlib import Path
from collections import defaultdict

class GradeOptDataLoader:
    def __init__(self, json_path, dataset_name="English_Grade12_004", max_score=15.0):
        self.json_path = Path(json_path)
        self.dataset_name = dataset_name
        self.max_score = float(max_score)
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

    def get_balanced_splits(self, all_txt_dir, q_id="66", train_size=None, val_size=None, val_ratio=0.25, sample_ratio=None):
        """
        分层抽取平衡数据集。
        train_size 和 val_size 建议设为 3 的倍数，以便各档位平分。
        """
        all_txt_dir = Path(all_txt_dir)
        # 使用 defaultdict 按档位分类存储样本
        tier_buckets = defaultdict(list)

        # 1. 扫描文件并按档位入桶
        # 兼容你的文件名格式: {id}_{q_id}_{score}.txt
        txt_files = list(all_txt_dir.glob(f"*_{q_id}_*.txt"))
        if not txt_files:
            # 兼容旧格式: {id}_{q_id}.txt
            txt_files = list(all_txt_dir.glob(f"*_{q_id}.txt"))

        for tf in txt_files:
            student_id, parsed_qid = self._parse_txt_stem(tf.stem)
            if not student_id or parsed_qid != str(q_id):
                print(f"Skipping unparsable OCR file: {tf.name}")
                continue
            true_score = self.score_lookup.get((student_id, str(q_id)))
            
            if true_score is not None:
                tier = self._get_tier(true_score, q_id)
                with open(tf, "r", encoding="utf-8") as f:
                    content = f.read().strip()

                    if "[OCR Error]" in content or len(content) < 10:
                        print(f"⏭️ Skipping invalid OCR file: {tf.name}")
                        continue
                    
                    tier_buckets[tier].append({
                        "id": student_id,
                        "text": content,
                        "true_score": true_score,
                        "tier": tier
                    })

        D_train, D_val = [], []

        if sample_ratio is not None:
            sample_ratio = float(sample_ratio)
            if not 0 < sample_ratio <= 1:
                raise ValueError(f"sample_ratio must be in (0, 1] when set; got {sample_ratio}")
            if sample_ratio < 1:
                original_total = sum(len(tier_buckets[tier]) for tier in ["Low", "Mid", "High"])
                for tier in ["Low", "Mid", "High"]:
                    samples = tier_buckets[tier]
                    if not samples:
                        continue
                    random.shuffle(samples)
                    keep_count = max(1, int(round(len(samples) * sample_ratio)))
                    tier_buckets[tier] = samples[:keep_count]
                downsampled_total = sum(len(tier_buckets[tier]) for tier in ["Low", "Mid", "High"])
                print(
                    f"🐞 Debug sample ratio active: ratio={sample_ratio:.4f}, "
                    f"usable={original_total}, downsampled={downsampled_total}"
                )

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
