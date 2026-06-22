import argparse
import json
import os
from pathlib import Path
import sys
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.env import load_env

load_env()

from utils_asro.data_loader import GradeOptDataLoader
from utils_asro.progress import log_progress
from engine import ASROEngine
from client import GradeOptClient

def load_yaml_config(config_path, section):
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到配置文件: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cfg = data.get(section)
    if cfg is None:
        raise KeyError(f"配置文件缺少 `{section}` 配置段")
    if not isinstance(cfg, dict):
        raise TypeError(f"`{section}` 配置段必须是 YAML mapping")
    return cfg


def _first_present_config(cfg, *keys):
    for key in keys:
        value = cfg.get(key)
        if value is not None:
            return value
    return None


def _ensure_training_ocr(cfg, data_dir, json_path, dataset_name, target_q):
    if not cfg.get("ocr", cfg.get("ocr_before_train", False)):
        log_progress("ocr", "pre-training OCR skipped", enabled=False)
        return
    data_path = Path(data_dir)
    log_progress("ocr", "pre-training OCR started", data_dir=data_path, target_q=target_q)

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from utils.ocr import run_ocr_for_directory

    def expected_txt_resolver(image_path):
        image_path = Path(image_path)
        parts = image_path.stem.split("_")
        if len(parts) >= 3 and parts[-2].isdigit() and len(parts[-2]) <= 3:
            qid = parts[-2]
        elif len(parts) >= 2 and parts[-1].isdigit() and len(parts[-1]) <= 3:
            qid = parts[-1]
        else:
            qid = None
        if qid != str(target_q):
            return "skip"
        return [image_path.with_suffix(".txt"), *image_path.parent.glob(f"{image_path.stem}_*.txt")]

    ocr_result = run_ocr_for_directory(
        input_dir=data_path,
        model_name=cfg.get("ocr_model"),
        api_key=cfg.get("api_key") or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=cfg.get("base_url", "https://openrouter.ai/api/v1"),
        force_ocr=cfg.get("force_ocr", False),
        workers=cfg.get("ocr_workers", cfg.get("max_workers", 5)),
        expected_txt_resolver=expected_txt_resolver,
        limit=_first_present_config(cfg, "limit", "ocr_limit", "ocr_num"),
    )
    log_progress(
        "ocr",
        "pre-training OCR finished",
        succeeded=len(ocr_result.get("succeeded", [])),
        skipped=len(ocr_result.get("skipped", [])),
        failed=len(ocr_result.get("failed", [])),
    )
    if ocr_result["failed"]:
        raise RuntimeError(
            f"OCR failed for {len(ocr_result['failed'])} training image(s). "
            f"See {data_path / 'error_report_ocr.csv'}."
        )


def _load_task_definition(json_path, dataset_name, target_q):
    """
        Prepares the initial G (question + grading rubrics + new rules, left empty for now)
    """
    log_progress("guideline", "loading task definition", dataset=dataset_name, target_q=target_q)
    fallback = {
        "Gqs": "",
        "Gsr": "",
        "Gar": "",
    }
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise RuntimeError(f"Failed to load dataset metadata from {json_path}") from exc

    dataset = data.get(dataset_name, {})
    subjective_questions = dataset.get("subjective_question", {})
    if isinstance(subjective_questions, dict):
        qinfo = subjective_questions.get(str(target_q), {})
        if isinstance(qinfo, dict):
            question = qinfo.get("question")
            rubric = qinfo.get("rubric")

            if question:
                fallback["Gqs"] = question if isinstance(question, str) else json.dumps(question, ensure_ascii=False, indent=2)
            if rubric:
                fallback["Gsr"] = rubric if isinstance(rubric, str) else json.dumps(rubric, ensure_ascii=False, indent=2)
            if not fallback["Gsr"]:
                raise ValueError(
                    f"Missing rubric for {dataset_name}/{target_q}. "
                    "ASRO training requires dataset.json subjective_question[qid].rubric."
                )
            return fallback
    raise ValueError(
        f"Missing subjective_question entry for {dataset_name}/{target_q}. "
        "ASRO training requires dataset.json subjective_question[qid].rubric."
    )


def main():
    parser = argparse.ArgumentParser(description="[AES] ASRO 规则进化优化器")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument("--section", type=str, default="asro_train", help="YAML 配置段名称，默认 asro_train")
    parser.add_argument("--num", type=int, default=None, help="Limit pre-training OCR to this many images, overriding OCR limit config.")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config, args.section)
    log_progress("startup", "config loaded", config=args.config, section=args.section)
    if args.num is not None:
        cfg["limit"] = args.num

    data_dir = cfg["data_dir"]
    json_path = cfg.get("json_path", "./data/dataset.json")
    dataset_name = cfg.get("dataset_name", "English_Grade12_004")
    target_q = str(cfg.get("target_q", "66"))
    train_n = cfg.get("train_n")
    val_n = cfg.get("val_n")
    val_ratio = cfg.get("val_ratio", 0.25)
    max_score = cfg.get("max_score", 15.0)
    tier_count = cfg.get("tier_count", 5)
    misconf_tier_weight = cfg.get("misconf_tier_weight", 10.0)
    T = cfg.get("T", 10)
    B = cfg.get("B", 4)
    K = cfg.get("K", 4)
    max_workers = cfg.get("max_workers", 5)
    output_dir = cfg.get("output_dir", ".")
    api_key = cfg.get("api_key")
    base_url = cfg.get("base_url", "https://openrouter.ai/api/v1")
    grader_model = cfg.get("grader_model", "deepseek/deepseek-v4-flash")
    reflector_model = cfg.get("reflector_model", "deepseek/deepseek-v4-pro")
    client_timeout = cfg.get("client_timeout", 60.0)
    grader_temperature = cfg.get("grader_temperature", 0.0)
    reflector_temperature = cfg.get("reflector_temperature", 0.7)
    grader_max_tokens = cfg.get("grader_max_tokens")
    reflector_max_tokens = cfg.get("reflector_max_tokens")
    reflector_timeout = cfg.get("reflector_timeout")
    debug = cfg.get("debug", False)
    debug_data_ratio = cfg.get("debug_data_ratio")
    sample_ratio = debug_data_ratio if debug else None
    sample_filter_enabled = cfg.get("sample_filter_enabled", False)
    sample_filter_model = cfg.get("sample_filter_model", grader_model)

    log_progress(
        "startup",
        "training parameters resolved",
        dataset=dataset_name,
        target_q=target_q,
        train_n=train_n,
        val_n=val_n,
        T=T,
        B=B,
        K=K,
        workers=max_workers,
        output_dir=output_dir,
        debug=debug,
        debug_data_ratio=sample_ratio,
        sample_filter_enabled=sample_filter_enabled,
        sample_filter_model=sample_filter_model,
    )

    _ensure_training_ocr(cfg, data_dir, json_path, dataset_name, target_q)

    initial_G = _load_task_definition(json_path, dataset_name, target_q)
    initial_G["max_score"] = max_score
    initial_G["tier_count"] = tier_count

    log_progress(
        "data",
        "creating data loader",
        json_path=json_path,
        dataset=dataset_name,
        sample_filter_enabled=sample_filter_enabled
    )
    loader = GradeOptDataLoader(
        json_path,
        dataset_name=dataset_name,
        max_score=max_score,
        sample_filter_enabled=sample_filter_enabled,
        sample_filter_model=sample_filter_model,
        api_key=api_key,
        base_url=base_url,
        timeout=client_timeout,
        max_workers=max_workers,
        task_context=initial_G,
    )

    log_progress("data", "building balanced train/val splits", data_dir=data_dir, target_q=target_q)
    D_train, D_val = loader.get_balanced_splits(
        data_dir, 
        q_id=target_q, 
        train_size=train_n, 
        val_size=val_n,
        val_ratio=val_ratio,
        sample_ratio=sample_ratio,
    )
    log_progress("data", "balanced splits ready", train=len(D_train), val=len(D_val))

    log_progress("client", "creating ASRO client", grader_model=grader_model, reflector_model=reflector_model)
    client = GradeOptClient(
        api_key=api_key,
        base_url=base_url,
        grader_model=grader_model,
        reflector_model=reflector_model,
        timeout=client_timeout,
        grader_temperature=grader_temperature,
        reflector_temperature=reflector_temperature,
        grader_max_tokens=grader_max_tokens,
        reflector_max_tokens=reflector_max_tokens,
        reflector_timeout=reflector_timeout,
    ) 
    log_progress("engine", "creating ASRO engine", T=T, B=B, K=K, workers=max_workers)
    engine = ASROEngine(
        client,
        T=T,
        B=B,
        K=K,
        max_score=max_score,
        tier_count=tier_count,
        misconf_tier_weight=misconf_tier_weight,
        max_workers=max_workers,
        output_dir=output_dir,
        debug=debug,
    )

    log_progress("optimization", "ASRO optimization started", rounds=T, train=len(D_train), val=len(D_val))
    best_g = engine.run_ASRO_optimization(D_train, D_val, initial_G) 

    train_label = len(D_train) if train_n is None else train_n
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"optimized_guideline_Q{target_q}_T{T}_B{B}_K{K}_W{max_workers}_N{train_label}.json")
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(best_g, f, ensure_ascii=False, indent=2)
    log_progress("complete", "optimized guideline saved", path=output_filename)
    
    print(f"🎊 优化完成！最佳评分规则已保存至: {output_filename}")

if __name__ == "__main__":
    main()
