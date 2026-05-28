from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main


class SmokeAutoScoreClient:
    def __init__(self, *args, **kwargs):
        pass

    def call_llm(self, system_prompt, user_prompt, **payload):
        return '{"total_score": 10.0, "feedback": "smoke test"}'

    def autoscore_grade(self, task_rubric, essay_text, tier, required_points=None):
        return {"total_score": float(tier) * 2.0, "feedback": "smoke test"}


def _sample_scan_files(scan_dir: Path, samples_per_paper: int, target_ids: set[str] | None):
    files = []
    for path in sorted(scan_dir.glob("*.tif")):
        student_id, q_id, _ = main._parse_essay_stem(path.stem)
        if not student_id or not q_id:
            continue
        if target_ids and q_id not in target_ids:
            continue
        files.append(path)
        if len(files) >= samples_per_paper:
            break
    return files


def _write_smoke_inputs(paper_dir: Path, work_root: Path, samples_per_paper: int, target_ids: set[str] | None):
    scan_dir = paper_dir / "Scan"
    if not scan_dir.exists():
        raise FileNotFoundError(f"Missing Scan directory: {scan_dir}")

    selected = _sample_scan_files(scan_dir, samples_per_paper, target_ids)
    if not selected:
        raise FileNotFoundError(f"No scan files selected from: {scan_dir}")

    input_dir = work_root / paper_dir.name
    input_dir.mkdir(parents=True, exist_ok=True)

    for idx, scan in enumerate(selected):
        student_id, q_id, _ = main._parse_essay_stem(scan.stem)
        score = 8.0 + (idx % 5)
        txt_path = input_dir / f"{student_id}_{q_id}_{score:.1f}.txt"
        txt_path.write_text(
            (
                "This is a local smoke-test essay for the ExamOCR pipeline. "
                "It is long enough to pass validation and exercises scoring, "
                "tiering, result collection, and metric calculation."
            ),
            encoding="utf-8",
        )

    return input_dir, selected


def run_smoke(data_root: Path, out_root: Path, samples_per_paper: int, target_id: str | None):
    if samples_per_paper < 5 or samples_per_paper > 10:
        raise ValueError("--samples-per-paper must be between 5 and 10")

    papers = sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("English_Grade12_"))
    if not papers:
        raise FileNotFoundError(f"No paper directories found under: {data_root}")

    target_ids = main._target_ids(target_id)
    work_root = out_root / "inputs"
    results_root = out_root / "results"
    if out_root.exists():
        shutil.rmtree(out_root)
    work_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    original_client = main.AutoScoreClient
    main.AutoScoreClient = SmokeAutoScoreClient
    try:
        for paper in papers:
            input_dir, selected = _write_smoke_inputs(paper, work_root, samples_per_paper, target_ids)
            result = main.run_autoscore_pipeline(
                input_dir=str(input_dir),
                out_dir=str(results_root / paper.name),
                json_path=str(data_root / "__not_used_in_smoke__.json"),
                dataset_name=paper.name,
                target_id=target_id,
                num=samples_per_paper,
                workers=1,
                mode="baseline",
                api_key="smoke-test-key",
                base_url="http://127.0.0.1/smoke",
                ocr_only=False,
                debug=False,
            )
            if result is None or len(result) != len(selected):
                actual = 0 if result is None else len(result)
                raise RuntimeError(f"{paper.name}: expected {len(selected)} rows, got {actual}")
            print(f"SMOKE_OK {paper.name} rows={len(result)}")
    finally:
        main.AutoScoreClient = original_client


def parse_args():
    parser = argparse.ArgumentParser(description="Local smoke test for the ExamOCR scoring pipeline.")
    parser.add_argument("--data-root", default="data/processed_5.0pct_66_67/processed_5.0pct_66_67", help="Directory containing English_Grade12_* paper folders.")
    parser.add_argument("--out-root", default=".smoke_test", help="Temporary smoke-test output directory.")
    parser.add_argument("--samples-per-paper", type=int, default=6, help="Pick 5-10 cases per paper.")
    parser.add_argument("--target-id", default="66,67", help="Question id filter, for example 66 or 66,67.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_smoke(
        data_root=Path(args.data_root),
        out_root=Path(args.out_root),
        samples_per_paper=args.samples_per_paper,
        target_id=args.target_id,
    )
