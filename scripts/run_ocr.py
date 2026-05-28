from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.env import load_env

load_env()

from utils.ocr import run_ocr_for_directory


def load_config(path):
    if not path:
        return {}
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "ocr" in data and isinstance(data["ocr"], dict):
        return data["ocr"]
    if "pipeline" in data and isinstance(data["pipeline"], dict):
        return data["pipeline"]
    return data


def first_present_config(cfg, *keys):
    for key in keys:
        value = cfg.get(key)
        if value is not None:
            return value
    return None


def main():
    parser = argparse.ArgumentParser(description="Run OCR for images in a directory.")
    parser.add_argument("--input_dir", required=True, help="Directory containing answer images. OCR .txt files are written next to each image.")
    parser.add_argument("--config", help="YAML config file. Reads `ocr` or `pipeline` section when present.")
    parser.add_argument("--model", dest="ocr_model", help="Override OCR model name.")
    parser.add_argument("--force_ocr", action="store_true", help="Regenerate txt files even if they exist.")
    parser.add_argument("--debug", action="store_true", help="Write side-by-side OCR debug images.")
    parser.add_argument("--limit", "--num", dest="limit", type=int, help="Maximum number of images to OCR.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    limit = args.limit
    if limit is None:
        limit = first_present_config(cfg, "limit", "ocr_limit", "ocr_num", "num")

    result = run_ocr_for_directory(
        input_dir=args.input_dir,
        model_name=args.ocr_model or cfg.get("ocr_model", "qwen/qwen-2-vl-72b-instruct"),
        api_key=cfg.get("api_key"),
        base_url=cfg.get("base_url", "https://openrouter.ai/api/v1"),
        prompt=cfg.get("prompt"),
        force_ocr=args.force_ocr or cfg.get("force_ocr", False),
        debug=args.debug or cfg.get("debug_ocr", cfg.get("debug", False)),
        workers=cfg.get("ocr_workers", cfg.get("workers", 1)),
        use_multithread=cfg.get("use_multithread", True),
        skip_existing=cfg.get("skip_existing", True),
        limit=limit,
    )
    print(f"OCR complete: processed={len(result['processed'])}, skipped={len(result['skipped'])}, failed={len(result['failed'])}")
    if result["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
