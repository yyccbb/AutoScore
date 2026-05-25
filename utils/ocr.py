from __future__ import annotations

import csv
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, Optional

_PROJECT_DIR = str(Path(__file__).resolve().parents[1])
_shadowing_paths = []
for _path in ("", _PROJECT_DIR):
    if _path in sys.path:
        sys.path.remove(_path)
        _shadowing_paths.append(_path)
try:
    from tqdm import tqdm
finally:
    for _path in reversed(_shadowing_paths):
        sys.path.insert(0, _path)

import utils.prompts as prompts


DEFAULT_EXTRA_HEADERS = {
    "HTTP-Referer": "https://github.com/OpenRouterTeam/AES",
    "X-Title": "AES_OCR",
}


def image_extensions():
    return (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


def perform_ocr_image(
    image_path,
    model_name,
    api_key=None,
    base_url="https://openrouter.ai/api/v1",
    prompt=None,
    temperature=0.01,
    timeout=60,
    extra_headers=None,
):
    image_path = Path(image_path)
    start_time = time.time()
    try:
        from utils.llm_api import call_llm

        text = call_llm(
            model_name=model_name,
            system_prompt=None,
            user_prompt=prompt or prompts.BASIC_OCR_PROMPT,
            images=[image_path],
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            timeout=timeout,
            extra_headers=extra_headers or DEFAULT_EXTRA_HEADERS,
        )
        return {
            "image_path": str(image_path),
            "text": (text or "").strip(),
            "success": True,
            "latency": time.time() - start_time,
            "error": "",
        }
    except Exception as exc:
        return {
            "image_path": str(image_path),
            "text": "",
            "success": False,
            "latency": time.time() - start_time,
            "error": str(exc),
        }


def create_comparison_image(img_path, ocr_text, output_path):
    from PIL import Image, ImageDraw, ImageFont

    img_path = Path(img_path)
    output_path = Path(output_path)
    try:
        orig_img = Image.open(img_path).convert("RGB")
    except Exception as exc:
        print(f"Warning: failed to open image for OCR debug comparison {img_path}: {exc}")
        return

    w, h = orig_img.size
    new_img = Image.new("RGB", (w * 2, h), "white")
    new_img.paste(orig_img, (0, 0))
    draw = ImageDraw.Draw(new_img)
    font_size = max(16, int(h / 60))

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    chars_per_line = int(w / (font_size * 0.6)) if font_size > 0 else 60
    lines = []
    for paragraph in str(ocr_text).split("\n"):
        if paragraph.strip() == "":
            lines.append("")
        else:
            lines.extend(textwrap.wrap(paragraph, width=max(1, chars_per_line)))

    draw.text((w + 20, 20), "\n".join(lines), fill="black", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_img.save(output_path)


def _resolver_indicates_existing(resolved):
    if resolved is None:
        return False
    if isinstance(resolved, bool):
        return resolved
    if isinstance(resolved, (str, Path)):
        return Path(resolved).exists()
    if isinstance(resolved, Iterable):
        return any(_resolver_indicates_existing(item) for item in resolved)
    return bool(resolved)


def _resolver_indicates_forced_skip(resolved):
    return isinstance(resolved, str) and resolved == "skip"


def _resolver_output_path(resolved):
    if resolved is None or isinstance(resolved, bool):
        return None
    if isinstance(resolved, (str, Path)):
        if _resolver_indicates_forced_skip(resolved):
            return None
        path = Path(resolved)
        return path if path.suffix.lower() == ".txt" else None
    if isinstance(resolved, Iterable):
        for item in resolved:
            path = _resolver_output_path(item)
            if path is not None:
                return path
    return None


def _has_existing_txt(image_path: Path, expected_txt_resolver: Optional[Callable]):
    if expected_txt_resolver is None:
        return image_path.with_suffix(".txt").exists()
    return _resolver_indicates_existing(expected_txt_resolver(image_path))


def _write_error_report(input_path: Path, failed):
    if not failed:
        return None
    report_path = input_path / "error_report_ocr.csv"
    with open(report_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "error", "latency"])
        writer.writeheader()
        for item in failed:
            writer.writerow(
                {
                    "image_path": item.get("image_path", ""),
                    "error": item.get("error", ""),
                    "latency": item.get("latency", 0),
                }
            )
    return report_path


def run_ocr_for_directory(
    input_dir,
    model_name,
    api_key=None,
    base_url="https://openrouter.ai/api/v1",
    prompt=None,
    force_ocr=False,
    debug=False,
    workers=1,
    use_multithread=True,
    skip_existing=True,
    expected_txt_resolver=None,
    limit=None,
):
    input_path = Path(input_dir)
    supported_exts = set(image_extensions())
    processed = []
    skipped = []
    failed = []

    images = [
        p
        for p in sorted(input_path.iterdir())
        if p.is_file()
        and p.suffix.lower() in supported_exts
        and not p.name.endswith("_cmp.jpg")
    ]

    to_process = []
    for image_path in images:
        resolved = expected_txt_resolver(image_path) if expected_txt_resolver is not None else None
        if _resolver_indicates_forced_skip(resolved):
            skipped.append({"image_path": str(image_path), "reason": "filtered"})
            continue
        existing_txt = (
            _resolver_indicates_existing(resolved)
            if expected_txt_resolver is not None
            else image_path.with_suffix(".txt").exists()
        )
        if skip_existing and not force_ocr and existing_txt:
            skipped.append({"image_path": str(image_path), "reason": "existing_txt"})
            continue
        if limit is not None and len(to_process) >= int(limit):
            skipped.append({"image_path": str(image_path), "reason": "limit"})
            continue
        to_process.append(image_path)

    print(
        f"OCR scan complete: total_images={len(images)}, "
        f"to_process={len(to_process)}, skipped={len(skipped)}, "
        f"force={bool(force_ocr)}, limit={limit}"
    )

    def process_one(image_path: Path):
        result = perform_ocr_image(
            image_path=image_path,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            prompt=prompt
        )
        if not result["success"]:
            return result

        try:
            resolved = expected_txt_resolver(image_path) if expected_txt_resolver is not None else None
            txt_out = _resolver_output_path(resolved) or image_path.with_suffix(".txt")
            txt_out.parent.mkdir(parents=True, exist_ok=True)
            with open(txt_out, "w", encoding="utf-8") as f:
                f.write(result["text"])
            result["txt_path"] = str(txt_out)
        except Exception as exc:
            result["success"] = False
            result["error"] = f"failed_to_write_txt: {exc}"
            return result

        if debug:
            cmp_out = image_path.with_name(f"{image_path.stem}_cmp.jpg")
            try:
                create_comparison_image(image_path, result["text"], cmp_out)
                result["comparison_path"] = str(cmp_out)
            except Exception as exc:
                result["comparison_error"] = str(exc)
        return result

    if use_multithread and workers and int(workers) > 1:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
            futures = {executor.submit(process_one, image_path): image_path for image_path in to_process}
            pbar = tqdm(total=len(futures), desc="OCR Scanning", ncols=70)
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as exc:
                    image_path = futures[future]
                    result = {
                        "image_path": str(image_path),
                        "text": "",
                        "success": False,
                        "latency": 0,
                        "error": str(exc),
                    }
                if result.get("success"):
                    processed.append(result)
                else:
                    failed.append(result)
                pbar.update(1)
            pbar.close()
    else:
        for image_path in tqdm(to_process, desc="OCR Scanning", ncols=70):
            try:
                result = process_one(image_path)
            except Exception as exc:
                result = {
                    "image_path": str(image_path),
                    "text": "",
                    "success": False,
                    "latency": 0,
                    "error": str(exc),
                }
            if result.get("success"):
                processed.append(result)
            else:
                failed.append(result)

    report_path = _write_error_report(input_path, failed)
    if failed:
        print(f"OCR failures: {len(failed)}. Error report: {report_path}")

    return {"processed": processed, "skipped": skipped, "failed": failed}
