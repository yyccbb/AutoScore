from datetime import datetime
import os
import json
import copy
import re
import numpy as np

def npx_converter(obj):
    """处理 Numpy 类型转 JSON"""
    if isinstance(obj, (np.int64, np.int32, np.int16)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


GRADER_TAG_RENDER_ORDER = (
    "SCORE",
    "TIER",
    "CORRECTED_MEANING",
    "TASK_REQUIREMENTS",
    "CONTENT_JUDGMENT",
    "LANGUAGE_JUDGMENT",
    "COHERENCE_JUDGMENT",
    "GAR_APPLICATION",
    "REASONING",
    "BOUNDARY_CHECK",
)

GRADER_LOG_METADATA_KEYS = {
    "id",
    "text",
    "true",
    "true_score",
    "pred",
    "misconf",
    "prob",
}


def format_grader_tags_for_log(sample):
    tag_names = [tag for tag in GRADER_TAG_RENDER_ORDER if tag in sample]
    tag_names.extend(
        sorted(
            key
            for key in sample
            if key not in GRADER_LOG_METADATA_KEYS
            and key not in GRADER_TAG_RENDER_ORDER
            and key.upper() == key
        )
    )
    if tag_names:
        chunks = []
        for tag_name in tag_names:
            chunks.append(f"{tag_name}:\n{str(sample.get(tag_name, '')).strip()}")
        return "\n\n".join(chunks).strip()
    return str(sample.get("reasoning", "No parsed grader tags provided"))


def save_guideline(guideline, round_idx, qwk_score, is_best=False, output_dir="optimized_guidelines"):
    save_dir = output_dir
    os.makedirs(save_dir, exist_ok=True)
    prefix = "BEST_" if is_best else ""
    filename = f"{prefix}round_{round_idx}_qwk_{qwk_score:.4f}"
    
    # 保存 JSON
    json_path = os.path.join(save_dir, f"{filename}.json")
    save_data = copy.deepcopy(guideline)
    save_data['round'] = int(round_idx)
    save_data['qwk'] = float(qwk_score)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=4, default=npx_converter)
    print(f"💾 Saved guideline to {json_path}")

def save_error_logs(results, round_idx, top_n=10):
    log_dir = "error_analysis_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 按 Misconf 排序取最惨的 N 个
    worst_samples = sorted(results, key=lambda x: x['misconf'], reverse=True)[:top_n]
    
    file_path = os.path.join(log_dir, f"round_{round_idx}_bad_cases.md")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"# 🚨 Round {round_idx} Error Analysis\n\n")
        f.write(f"> Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for i, s in enumerate(worst_samples):
            f.write(f"## Case {i+1} | Misconf: {s['misconf']:.2f}\n")
            # ✨ 在这里加入 ID 展示
            f.write(f"- **ID:** `{s['id']}`\n") 
            f.write(f"- **True:** {s['true']} | **Pred:** {s['pred']}\n")
            f.write(f"### 📝 Essay:\n> {s['text']}\n\n")
            f.write(f"### 🧠 Grader Tags:\n```text\n{format_grader_tags_for_log(s)}\n```\n")
            f.write(f"\n---\n")
            
    print(f"📝 Bad cases saved to {file_path}")

def print_round_dashboard(round_idx, results, qwk):
    probs = [r.get('prob', 0.5) for r in results]
    sorted_samples = sorted(results, key=lambda x: x['misconf'], reverse=True)
    
    print(f"\n" + "="*70)
    print(f"📈 [ROUND {round_idx} DASHBOARD]")
    print(f"✅ QWK (Current): {qwk:.4f} | 🧠 Avg Confidence: {np.mean(probs):.2f}")
    print(f"\n🔍 Top-5 Problematic Samples:")
    print(f"{'ID':>12} | {'True':>6} | {'Pred':>6} | {'Misconf':>10}")
    for s in sorted_samples[:5]:
        sample_id = str(s.get("id", "unknown"))
        print(f"{sample_id:>12} | {s['true']:6.1f} | {s['pred']:6.1f} | {s['misconf']:10.2f}")
    print("="*70 + "\n")

def _score_to_tier(score, max_score=15.0, tier_count=5):
    """Convert a score to a 1-based tier using configurable equal-width bins."""
    if score <= 0:
        return 0
    step = float(max_score) / float(tier_count)
    return min(tier_count, max(1, int(np.ceil(float(score) / step))))


def _legacy_cleanup(raw_text):
    """
    专门清理 LLM (尤其是 DeepSeek-R1) 输出的工业级清洗器
    """
    if not raw_text:
        return ""
    
    # 1. 去掉 DeepSeek 的思考过程 <think>...</think>
    text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
    
    # 2. 去掉 Markdown 的代码块包裹 (如 ```markdown ... ``` 或 ```)
    # 匹配 ```任意字符 [换行] 内容 [换行] ```
    text = re.sub(r'^```[a-zA-Z]*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?```$', '', text, flags=re.MULTILINE)
    
    # 3. 去掉前后多余的空白
    return text.strip()
