from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wasserstein_distance
from sklearn.metrics import cohen_kappa_score


DRAW_COLUMNS = [
    "student_id",
    "q_id",
    "true_score",
    "ai_score",
    "diff",
    "abs_error",
    "is_anomaly",
    "tier",
    "stem",
    "draw_index",
    "is_non_finite",
    "is_out_of_range",
]

SUMMARY_COLUMNS = [
    "student_id",
    "q_id",
    "stem",
    "true_score",
    "draws_requested",
    "draws_succeeded",
    "valid_scores",
    "failed_draws",
    "ai_mean",
    "ai_median",
    "ai_std",
    "ai_min",
    "ai_max",
    "ai_range",
    "unique_scores",
    "mean_bias",
    "mean_abs_error",
    "out_of_range_scores",
]


def _finite_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric[np.isfinite(numeric)]


def _describe(values: pd.Series) -> dict:
    values = _finite_numeric(values)
    if values.empty:
        return {
            "n": 0,
            "mean": math.nan,
            "std": math.nan,
            "min": math.nan,
            "q1": math.nan,
            "median": math.nan,
            "q3": math.nan,
            "max": math.nan,
        }
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else math.nan,
        "min": float(values.min()),
        "q1": float(values.quantile(0.25)),
        "median": float(values.median()),
        "q3": float(values.quantile(0.75)),
        "max": float(values.max()),
    }


def _paired_metrics(true_values: pd.Series, predicted_values: pd.Series) -> dict:
    pairs = pd.DataFrame({"true": true_values, "pred": predicted_values}).apply(
        pd.to_numeric, errors="coerce"
    )
    pairs = pairs[np.isfinite(pairs["true"]) & np.isfinite(pairs["pred"])]
    if pairs.empty:
        return {
            "n": 0,
            "bias": math.nan,
            "mae": math.nan,
            "rmse": math.nan,
            "qwk": math.nan,
            "pearson_r": math.nan,
            "spearman_rho": math.nan,
            "wasserstein": math.nan,
        }

    true = pairs["true"].to_numpy(dtype=float)
    pred = pairs["pred"].to_numpy(dtype=float)
    diff = pred - true

    true_labels = np.rint(true * 2).astype(int)
    pred_labels = np.rint(pred * 2).astype(int)
    qwk = math.nan
    if len(pairs) >= 2 and len(np.unique(np.concatenate([true_labels, pred_labels]))) >= 2:
        try:
            qwk = float(
                cohen_kappa_score(
                    true_labels,
                    pred_labels,
                    weights="quadratic",
                )
            )
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    pearson = math.nan
    spearman = math.nan
    if len(pairs) >= 2 and np.ptp(true) > 0 and np.ptp(pred) > 0:
        pearson = float(stats.pearsonr(true, pred).statistic)
        spearman = float(stats.spearmanr(true, pred).statistic)

    return {
        "n": int(len(pairs)),
        "bias": float(np.mean(diff)),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(np.square(diff)))),
        "qwk": qwk,
        "pearson_r": pearson,
        "spearman_rho": spearman,
        "wasserstein": float(wasserstein_distance(true, pred)),
    }


def _mean_bias_confidence_interval(true_values: pd.Series, predicted_values: pd.Series):
    pairs = pd.DataFrame({"true": true_values, "pred": predicted_values}).apply(
        pd.to_numeric, errors="coerce"
    )
    pairs = pairs[np.isfinite(pairs["true"]) & np.isfinite(pairs["pred"])]
    diff = (pairs["pred"] - pairs["true"]).to_numpy(dtype=float)
    if len(diff) < 2:
        return math.nan, math.nan
    if np.ptp(diff) == 0:
        value = float(diff[0])
        return value, value
    interval = stats.t.interval(
        0.95,
        df=len(diff) - 1,
        loc=float(np.mean(diff)),
        scale=float(stats.sem(diff)),
    )
    return float(interval[0]), float(interval[1])


def _format_number(value) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "N/A"
    return f"{number:.4f}"


def _markdown_metric_table(title: str, metrics: dict) -> list[str]:
    labels = {
        "n": "Valid pairs",
        "bias": "Mean bias (AI - truth)",
        "mae": "MAE",
        "rmse": "RMSE",
        "qwk": "Quadratic weighted kappa",
        "pearson_r": "Pearson r",
        "spearman_rho": "Spearman rho",
        "wasserstein": "Wasserstein distance",
    }
    lines = [f"### {title}", "", "| Metric | Value |", "| --- | ---: |"]
    for key, label in labels.items():
        value = metrics.get(key)
        rendered = str(int(value)) if key == "n" else _format_number(value)
        lines.append(f"| {label} | {rendered} |")
    lines.append("")
    return lines


def _markdown_descriptive_table(rows: list[tuple[str, dict]]) -> list[str]:
    lines = [
        "### Distribution summaries",
        "",
        "| Distribution | N | Mean | Std | Min | Q1 | Median | Q3 | Max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, values in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    str(values["n"]),
                    _format_number(values["mean"]),
                    _format_number(values["std"]),
                    _format_number(values["min"]),
                    _format_number(values["q1"]),
                    _format_number(values["median"]),
                    _format_number(values["q3"]),
                    _format_number(values["max"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _prepare_draws(draws: pd.DataFrame, max_score: float) -> pd.DataFrame:
    prepared = draws.copy()
    for column in DRAW_COLUMNS:
        if column not in prepared.columns:
            prepared[column] = pd.Series(dtype="object")
    prepared["ai_score"] = pd.to_numeric(prepared["ai_score"], errors="coerce")
    prepared["true_score"] = pd.to_numeric(prepared["true_score"], errors="coerce")
    prepared["is_non_finite"] = ~np.isfinite(prepared["ai_score"])
    prepared["is_out_of_range"] = (
        np.isfinite(prepared["ai_score"])
        & ((prepared["ai_score"] < 0) | (prepared["ai_score"] > float(max_score)))
    )
    prepared = prepared[DRAW_COLUMNS]
    if not prepared.empty:
        prepared = prepared.sort_values(
            ["q_id", "student_id", "draw_index"], kind="stable"
        ).reset_index(drop=True)
    return prepared


def _build_essay_summary(
    draws: pd.DataFrame,
    expected_essays: pd.DataFrame,
    samples_per_essay: int,
    max_score: float,
) -> pd.DataFrame:
    rows = []
    for expected in expected_essays.to_dict("records"):
        student_id = str(expected["student_id"])
        q_id = str(expected["q_id"])
        matching = draws[
            (draws["student_id"].astype(str) == student_id)
            & (draws["q_id"].astype(str) == q_id)
        ]
        scores = _finite_numeric(matching["ai_score"])
        true_score = float(expected["true_score"])
        ai_mean = float(scores.mean()) if not scores.empty else math.nan
        ai_min = float(scores.min()) if not scores.empty else math.nan
        ai_max = float(scores.max()) if not scores.empty else math.nan
        rows.append(
            {
                "student_id": student_id,
                "q_id": q_id,
                "stem": expected.get("stem", ""),
                "true_score": true_score,
                "draws_requested": int(samples_per_essay),
                "draws_succeeded": int(len(matching)),
                "valid_scores": int(len(scores)),
                "failed_draws": int(samples_per_essay - len(matching)),
                "ai_mean": ai_mean,
                "ai_median": float(scores.median()) if not scores.empty else math.nan,
                "ai_std": float(scores.std(ddof=1)) if len(scores) > 1 else math.nan,
                "ai_min": ai_min,
                "ai_max": ai_max,
                "ai_range": ai_max - ai_min if not scores.empty else math.nan,
                "unique_scores": int(scores.nunique()),
                "mean_bias": ai_mean - true_score if np.isfinite(ai_mean) else math.nan,
                "mean_abs_error": abs(ai_mean - true_score) if np.isfinite(ai_mean) else math.nan,
                "out_of_range_scores": int(
                    ((scores < 0) | (scores > float(max_score))).sum()
                ),
            }
        )
    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    if not summary.empty:
        summary = summary.sort_values(["q_id", "student_id"], kind="stable").reset_index(
            drop=True
        )
    return summary


def _plot_distributions(
    draws: pd.DataFrame,
    summary: pd.DataFrame,
    output_path: Path,
    dataset_name: str,
    max_score: float,
):
    question_ids = sorted(summary["q_id"].astype(str).unique())
    if not question_ids:
        question_ids = ["unknown"]
    fig, axes = plt.subplots(
        len(question_ids),
        2,
        figsize=(13, max(4.5, 4.2 * len(question_ids))),
        squeeze=False,
    )

    for row_idx, q_id in enumerate(question_ids):
        q_summary = summary[summary["q_id"].astype(str) == q_id]
        q_draws = draws[draws["q_id"].astype(str) == q_id]
        truth = _finite_numeric(q_summary["true_score"])
        all_draws = _finite_numeric(q_draws["ai_score"])
        essay_means = _finite_numeric(q_summary["ai_mean"])
        combined = pd.concat([truth, all_draws, essay_means], ignore_index=True)
        low_value = min(0.0, float(combined.min())) if not combined.empty else 0.0
        high_value = max(float(max_score), float(combined.max())) if not combined.empty else float(max_score)
        low_edge = math.floor(low_value * 2) / 2 - 0.25
        high_edge = math.ceil(high_value * 2) / 2 + 0.75
        bins = np.arange(low_edge, high_edge + 0.5, 0.5)

        comparisons = [
            (axes[row_idx][0], all_draws, "All AI draws"),
            (axes[row_idx][1], essay_means, "AI essay means"),
        ]
        for axis, ai_values, ai_label in comparisons:
            if not truth.empty:
                axis.hist(
                    truth,
                    bins=bins,
                    density=True,
                    alpha=0.45,
                    color="#4C78A8",
                    label="Ground truth",
                )
            if not ai_values.empty:
                axis.hist(
                    ai_values,
                    bins=bins,
                    density=True,
                    alpha=0.45,
                    color="#F58518",
                    label=ai_label,
                )
            axis.set_xlim(low_edge, high_edge)
            axis.set_xlabel("Score")
            axis.set_ylabel("Density")
            axis.set_title(f"Question {q_id}: ground truth vs {ai_label.lower()}")
            axis.grid(alpha=0.2)
            axis.legend()

    fig.suptitle(f"Baseline score distributions: {dataset_name}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_baseline_distribution_analysis(
    draws: pd.DataFrame,
    expected_essays: pd.DataFrame,
    out_dir,
    dataset_name: str,
    samples_per_essay: int,
    max_score: float,
    model_name: str,
    temperature: float,
):
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    draws_path = output_dir / f"baseline_draws_{dataset_name}.csv"
    summary_path = output_dir / f"baseline_essay_summary_{dataset_name}.csv"
    report_path = output_dir / f"baseline_distribution_report_{dataset_name}.md"
    plot_path = output_dir / f"baseline_distribution_{dataset_name}.png"

    prepared_draws = _prepare_draws(draws, max_score=max_score)
    summary = _build_essay_summary(
        prepared_draws,
        expected_essays=expected_essays,
        samples_per_essay=samples_per_essay,
        max_score=max_score,
    )
    prepared_draws.to_csv(draws_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    report = [
        f"# Baseline Distribution Report: {dataset_name}",
        "",
        f"- Model: `{model_name}`",
        f"- Temperature: `{temperature}`",
        f"- Requested samples per essay: `{samples_per_essay}`",
        f"- Expected score range: `0` to `{max_score}`",
        "",
    ]

    question_ids = sorted(summary["q_id"].astype(str).unique())
    for q_id in question_ids:
        q_summary = summary[summary["q_id"].astype(str) == q_id]
        q_draws = prepared_draws[prepared_draws["q_id"].astype(str) == q_id]
        valid_draws = q_draws[~q_draws["is_non_finite"]]
        valid_means = q_summary[np.isfinite(q_summary["ai_mean"])]
        expected_calls = int(len(q_summary) * samples_per_essay)
        successful_calls = int(len(q_draws))
        valid_calls = int(len(valid_draws))

        report.extend(
            [
                f"## Question {q_id}",
                "",
                "### Completion diagnostics",
                "",
                f"- Essays: {len(q_summary)}",
                f"- Requested calls: {expected_calls}",
                f"- Successful parsed calls: {successful_calls}",
                f"- Failed calls: {expected_calls - successful_calls}",
                f"- Non-finite scores: {successful_calls - valid_calls}",
                f"- Incomplete essays: {int((q_summary['draws_succeeded'] < samples_per_essay).sum())}",
                f"- Essays with zero successful calls: {int((q_summary['draws_succeeded'] == 0).sum())}",
                f"- Out-of-range scores: {int(q_draws['is_out_of_range'].sum())}",
                "",
            ]
        )

        report.extend(
            _markdown_descriptive_table(
                [
                    ("Ground truth (one per essay)", _describe(q_summary["true_score"])),
                    ("All valid AI draws", _describe(valid_draws["ai_score"])),
                    ("AI essay means", _describe(valid_means["ai_mean"])),
                ]
            )
        )
        report.extend(
            _markdown_metric_table(
                "Draw-level accuracy",
                _paired_metrics(valid_draws["true_score"], valid_draws["ai_score"]),
            )
        )
        essay_metrics = _paired_metrics(valid_means["true_score"], valid_means["ai_mean"])
        report.extend(_markdown_metric_table("Essay-mean accuracy", essay_metrics))
        ci_low, ci_high = _mean_bias_confidence_interval(
            valid_means["true_score"], valid_means["ai_mean"]
        )
        repeatable = q_summary[q_summary["valid_scores"] >= 2]
        report.extend(
            [
                "### Essay-mean uncertainty and repeatability",
                "",
                f"- 95% confidence interval for mean bias: [{_format_number(ci_low)}, {_format_number(ci_high)}]",
                f"- Essays with at least two valid draws: {len(repeatable)}",
                f"- Mean within-essay standard deviation: {_format_number(repeatable['ai_std'].mean())}",
                f"- Median within-essay standard deviation: {_format_number(repeatable['ai_std'].median())}",
                f"- Mean within-essay score range: {_format_number(repeatable['ai_range'].mean())}",
                "",
            ]
        )

    report_path.write_text("\n".join(report), encoding="utf-8")
    _plot_distributions(
        prepared_draws,
        summary,
        output_path=plot_path,
        dataset_name=dataset_name,
        max_score=max_score,
    )
    return {
        "draws": prepared_draws,
        "summary": summary,
        "draws_path": draws_path,
        "summary_path": summary_path,
        "report_path": report_path,
        "plot_path": plot_path,
    }
