import copy
import csv
import json
import math
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_PROJECT_DIR = str(Path(__file__).resolve().parents[1])
_shadowing_paths = []
for _path in ("", _PROJECT_DIR):
    if _path in sys.path:
        sys.path.remove(_path)
        _shadowing_paths.append(_path)
try:
    import numpy as np
    from sklearn.metrics import cohen_kappa_score
    from tqdm import tqdm
finally:
    for _path in reversed(_shadowing_paths):
        sys.path.insert(0, _path)

from utils_asro.sampler import ASROSampler
from utils_asro.progress import log_progress
from utils_asro.utils import print_round_dashboard, save_guideline, _legacy_cleanup, _score_to_tier


class ASROEngine:
    def __init__(
        self,
        client,
        T=3,
        B=2,
        K=2,
        Lambda=0.5,
        max_score=15.0,
        tier_count=5,
        misconf_tier_weight=10.0,
        max_workers=1,
        output_dir=".",
    ):
        self.client = client
        self.T, self.B, self.K, self.Lambda = T, B, K, Lambda
        self.max_score = float(max_score)
        self.tier_count = int(tier_count)
        self.misconf_tier_weight = float(misconf_tier_weight)
        self.max_workers = max(1, int(max_workers))
        self.output_dir = output_dir
        self.failed_results = []

        self.sampler = ASROSampler(
            max_score=self.max_score,
            tier_count=self.tier_count,
            misconf_tier_weight=self.misconf_tier_weight,
            max_workers=self.max_workers,
        )
        from optimizer import GradeOptimizer

        self.optimizer = GradeOptimizer(client, output_dir=self.output_dir)

    def _validate_inputs(self, D_train, D_val, initial_G):
        if not D_train:
            raise ValueError("D_train is empty; ASRO optimization requires training samples.")
        if not D_val:
            raise ValueError("D_val is empty; ASRO optimization requires validation samples.")
        if not isinstance(initial_G, dict):
            raise ValueError("initial_G must be a dict.")
        if "Gar" not in initial_G:
            raise ValueError("initial_G must contain a Gar field.")

    def _coerce_valid_score(self, score, sample_id):
        try:
            score = float(score)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Predicted score for {sample_id} is not a float: {score!r}") from exc
        if not 0 <= score <= self.max_score:
            raise ValueError(f"Predicted score for {sample_id} is out of range [0, {self.max_score}]: {score}")
        return score

    def _failed_result(self, sample, exc, stage):
        return {
            "id": sample.get("id", "unknown") if isinstance(sample, dict) else "unknown",
            "true": sample.get("true_score") if isinstance(sample, dict) else None,
            "stage": stage,
            "error": str(exc),
            "text": sample.get("text", "") if isinstance(sample, dict) else "",
        }

    def _call_llm_compat(self, system_prompt, user_prompt, **payload):
        call_llm = getattr(self.client, "call_llm")
        code = getattr(call_llm, "__code__", None)
        varnames = code.co_varnames[: code.co_argcount] if code else ()
        if len(varnames) >= 3 and varnames[1:3] == ("prompt", "is_reflector"):
            prompt = f"{system_prompt}\n\n{user_prompt}".strip()
            return call_llm(prompt, is_reflector=payload.get("is_reflector", True))
        payload = {k: v for k, v in payload.items() if k != "is_reflector"}
        return call_llm(system_prompt, user_prompt, **payload)

    def summarize_eval_results(self, results):
        if not results:
            raise RuntimeError("Cannot summarize empty evaluation results.")

        y_true = np.array([float(r["true"]) for r in results], dtype=float)
        y_pred = np.array([float(r["pred"]) for r in results], dtype=float)
        abs_err = np.abs(y_pred - y_true)
        sq_err = (y_pred - y_true) ** 2
        y_true_qwk = [int(round(x * 2)) for x in y_true]
        y_pred_qwk = [int(round(x * 2)) for x in y_pred]

        qwk = float(cohen_kappa_score(y_true_qwk, y_pred_qwk, weights="quadratic"))
        if not np.isfinite(qwk):
            qwk = 0.0

        return {
            "qwk": qwk,
            "mae": float(np.mean(abs_err)),
            "rmse": float(np.sqrt(np.mean(sq_err))),
            "avg_misconf": float(np.mean([float(r["misconf"]) for r in results])),
            "within_1": float(np.mean(abs_err <= 1.0)),
            "within_2": float(np.mean(abs_err <= 2.0)),
            "large_error_rate": float(np.mean(abs_err >= 3.0)),
            "mean_bias": float(np.mean(y_pred - y_true)),
        }

    def _save_round_metrics(self, metrics):
        os.makedirs(self.output_dir, exist_ok=True)

        json_path = os.path.join(self.output_dir, f"round_{int(metrics['round'])}_metrics.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        csv_path = os.path.join(self.output_dir, "metrics.csv")
        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)

    def _save_intermediate_records(self, round_idx: int, dataset: list, llm_response: list, is_validation: bool):
        subdir_path = os.path.join(self.output_dir, f"rnd{round_idx}")
        os.makedirs(subdir_path, exist_ok=True)

        val_prefix = "val_" if is_validation else ""
        dataset_save_path = os.path.join(subdir_path, val_prefix + "dataset.json")
        llm_response_save_path = os.path.join(subdir_path, val_prefix + "llm.json")

        with open(dataset_save_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)

        with open(llm_response_save_path, "w", encoding="utf-8") as f:
            json.dump(llm_response, f, ensure_ascii=False, indent=4)

    def _process_single_mode(self, mode, p_current, scan_results, global_cm_str, round_idx=1):
        try:
            safe_mode = tuple(int(x) for x in mode)
            e_ij = [r for r in scan_results if self._is_mode(r, safe_mode)]
            e_plus_i, e_plus_j = self._get_contrastive_examples(scan_results, safe_mode)
            log_progress(
                "mode",
                "mode repair started",
                round=round_idx,
                mode=f"{safe_mode[0] / 2.0}->{safe_mode[1] / 2.0}",
                error_examples=len(e_ij),
                correct_true=len(e_plus_i),
                correct_pred=len(e_plus_j),
            )

            self.optimizer.stage = "reflecting"
            log_progress("reflector", "reflector step started", round=round_idx, mode=f"{safe_mode[0] / 2.0}->{safe_mode[1] / 2.0}")
            diag = self.optimizer.reflector_step(
                p_current,
                e_ij,
                {"target_true_examples": e_plus_i, "target_pred_examples": e_plus_j},
                safe_mode,
                global_cm_str,
                curr_round=round_idx,
            )
            log_progress("reflector", "reflector step finished", round=round_idx, mode=f"{safe_mode[0] / 2.0}->{safe_mode[1] / 2.0}")

            self.optimizer.stage = "refining"
            log_progress("refiner", "refiner step started", round=round_idx, mode=f"{safe_mode[0] / 2.0}->{safe_mode[1] / 2.0}")
            refined_gar = self.optimizer.refiner_step(p_current, diag, [], safe_mode, curr_round=round_idx)
            log_progress("refiner", "refiner step finished", round=round_idx, mode=f"{safe_mode[0] / 2.0}->{safe_mode[1] / 2.0}")

            p_new = copy.deepcopy(p_current)
            p_new["Gar"] = refined_gar
            p_new["target_mode"] = safe_mode
            return p_new
        except Exception as exc:
            log_progress("mode", "mode repair failed", round=round_idx, mode=mode, error=exc)
            print(f"[WARN] Mode {mode} optimization failed: {exc}")
            return {
                "failed": True,
                "stage": getattr(exc, "stage", "optimizer"),
                "mode": list(mode) if isinstance(mode, tuple) else mode,
                "error": str(exc),
                "raw_response": getattr(exc, "raw_response", None),
            }

    def calculate_kappa_sequential(self, guideline, dataset):
        y_true = [int(round(s["true_score"] * 2)) for s in dataset]
        y_pred = []
        for sample in tqdm(dataset, desc="Kappa Eval", leave=False, ncols=70):
            score, _, _, _ = self.client.get_ordinal_score(sample["text"], guideline)
            score = self._coerce_valid_score(score, sample.get("id", "unknown"))
            y_pred.append(int(round(score * 2)))
        return cohen_kappa_score(y_true, y_pred, weights="quadratic")

    def _evaluate_sample(self, sample, guideline):
        score, _, _, reasoning = self.client.get_ordinal_score(sample["text"], guideline, sample["true_score"])
        score = self._coerce_valid_score(score, sample.get("id", "unknown"))

        true_score = float(sample["true_score"])
        true_tier = _score_to_tier(true_score, self.max_score, self.tier_count)
        pred_tier = _score_to_tier(score, self.max_score, self.tier_count)
        misconf = ((true_score - score) ** 2) + self.misconf_tier_weight * (abs(true_tier - pred_tier) ** 2)

        return {
            "id": sample.get("id", "unknown"),
            "true": true_score,
            "pred": score,
            "misconf": misconf,
            "text": sample["text"],
            "reasoning": reasoning,
        }

    def _validate_sample(self, sample, guideline):
        score, _, _, reasoning = self.client.get_ordinal_score(sample["text"], guideline, sample["true_score"])
        score = self._coerce_valid_score(score, sample.get("id", "unknown"))

        true_score = float(sample["true_score"])
        true_tier = _score_to_tier(true_score, self.max_score, self.tier_count)
        pred_tier = _score_to_tier(score, self.max_score, self.tier_count)
        misconf = ((true_score - score) ** 2) + self.misconf_tier_weight * (abs(true_tier - pred_tier) ** 2)

        return {
            "id": sample.get("id", "unknown"),
            "true": true_score,
            "pred": score,
            "misconf": misconf,
            "reasoning": reasoning,
        }

    def _evaluate_collection(self, samples, guideline, worker_fn, desc, stage):
        results, failed_results = [], []
        log_progress(stage, "evaluation started", samples=len(samples), workers=self.max_workers)

        if self.max_workers <= 1:
            for sample in tqdm(samples, desc=desc, ncols=70):
                try:
                    results.append(worker_fn(sample, guideline))
                except Exception as exc:
                    failed_results.append(self._failed_result(sample, exc, stage))
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(worker_fn, sample, guideline): sample for sample in samples}
                pbar = tqdm(total=len(futures), desc=desc, ncols=70)
                for future in as_completed(futures):
                    sample = futures[future]
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        failed_results.append(self._failed_result(sample, exc, stage))
                    pbar.update(1)
                pbar.close()

        self.failed_results = failed_results
        log_progress(stage, "evaluation finished", ok=len(results), failed=len(failed_results))
        if not results:
            os.makedirs(self.output_dir, exist_ok=True)
            failed_path = os.path.join(self.output_dir, f"failed_{stage}.json")
            with open(failed_path, "w", encoding="utf-8") as f:
                json.dump(failed_results, f, ensure_ascii=False, indent=2)

            print(f"[ERROR] No valid {stage} results. failed_count={len(failed_results)}")
            for item in failed_results[:5]:
                print(f"- id={item.get('id')} true={item.get('true')} stage={item.get('stage')}")
                print(f"  error={item.get('error')}")
                print(f"  text={item.get('text', '')[:200]}")

            raise RuntimeError(
                f"No valid {stage} results. failed_count={len(failed_results)}. "
                f"See {failed_path}"
            )
        return results

    def evaluate_minibatch_sequential(self, minibatch, guideline):
        return self._evaluate_collection(
            minibatch,
            guideline,
            self._evaluate_sample,
            "Evaluating Minibatch",
            "minibatch_eval",
        )

    def evaluate_validation_sequential(self, guideline, dataset):
        return self._evaluate_collection(
            dataset,
            guideline,
            self._validate_sample,
            "Validating",
            "validation_eval",
        )

    def run_ASRO_optimization(self, D_train, D_val, initial_G):
        self._validate_inputs(D_train, D_val, initial_G)
        log_progress("optimization", "input validation passed", train=len(D_train), val=len(D_val), rounds=self.T)

        current_guideline = copy.deepcopy(initial_G)
        best_guideline = copy.deepcopy(initial_G)
        best_overall_misconf = None
        current_misconf = None
        temperature = 1.0
        alpha = 0.9

        for round_idx in range(1, self.T + 1):
            log_progress("round", "round started", round=round_idx, total_rounds=self.T, temperature=f"{temperature:.4f}")

            log_progress("sampling", "minibatch sampling started", round=round_idx, train=len(D_train))
            minibatch, full_llm_response = self.sampler.sample_minibatch(D_train, current_guideline, self.client)
            self._save_intermediate_records(round_idx, D_train, full_llm_response, is_validation=False)
            log_progress("sampling", "minibatch sampling finished", round=round_idx, minibatch=len(minibatch))

            scan_results = self.evaluate_minibatch_sequential(minibatch, current_guideline)
            minibatch_failed_count = len(self.failed_results)

            error_modes = self._get_top_k_modes(scan_results, self.K) # accumulated misconf
            log_progress(
                "modes",
                "top error modes selected",
                round=round_idx,
                count=len(error_modes),
                modes=json.dumps([list(mode) for mode in error_modes]),
            )
            if not error_modes:
                print("[OK] No major errors found. Skipping repair.")
                summary = self.summarize_eval_results(scan_results)
                metrics = {
                    "round": round_idx,
                    **summary,
                    "accepted": False,
                    "temperature": float(temperature),
                    "target_modes": "",
                    "failed_count": int(minibatch_failed_count),
                }
                self._save_round_metrics(metrics)
                log_progress("round", "round finished without repair", round=round_idx, metrics_saved=True)
                temperature *= alpha
                continue

            global_cm_str = self._generate_cm_report(scan_results)
            log_progress("modes", "confusion report generated", round=round_idx, report=global_cm_str or "none")
            new_candidate_pool = []
            optimizer_failed_results = []
            for mode in error_modes:
                candidate = self._process_single_mode(mode, current_guideline, scan_results, global_cm_str, round_idx)
                if candidate and candidate.get("failed"):
                    optimizer_failed_results.append(candidate)
                elif candidate:
                    new_candidate_pool.append(candidate)

            p_full = copy.deepcopy(current_guideline)
            log_progress("consolidate", "priority consolidation started", round=round_idx, candidates=len(new_candidate_pool))
            p_full["Gar"] = self.priority_consolidate(new_candidate_pool, current_guideline)
            p_full["target_mode"] = "FULL_REPAIR"
            log_progress("consolidate", "priority consolidation finished", round=round_idx)

            validation_results = self.evaluate_validation_sequential(p_full, D_val)
            self._save_intermediate_records(round_idx, D_val, validation_results, is_validation=True)
            validation_failed_count = len(self.failed_results)
            summary = self.summarize_eval_results(validation_results)
            new_misconf = summary["avg_misconf"]
            current_qwk = summary["qwk"]

            print_round_dashboard(round_idx, validation_results, current_qwk)

            delta_e = new_misconf - (current_misconf if current_misconf is not None else new_misconf)
            accept_new = False
            if delta_e <= 0:
                accept_new = True
            else:
                p_accept = math.exp(-delta_e / (max(temperature, 1e-6) * 2.0))
                if random.random() < p_accept:
                    accept_new = True
                    print(f"[SA] Accepted worse candidate (delta={delta_e:.2f}, p={p_accept:.2%})")

            if accept_new:
                current_guideline = copy.deepcopy(p_full)
                current_misconf = new_misconf

                if best_overall_misconf is None or new_misconf < best_overall_misconf:
                    best_overall_misconf = new_misconf
                    best_guideline = copy.deepcopy(p_full)
                    save_guideline(best_guideline, round_idx, new_misconf, is_best=True, output_dir=self.output_dir)
                    print(f"[OK] New best guideline: {new_misconf:.4f}")

            metrics = {
                "round": round_idx,
                **summary,
                "accepted": bool(accept_new),
                "temperature": float(temperature),
                "target_modes": json.dumps([list(mode) for mode in error_modes]),
                "failed_count": int(minibatch_failed_count + validation_failed_count + len(optimizer_failed_results)),
            }
            self._save_round_metrics(metrics)
            log_progress(
                "round",
                "round finished",
                round=round_idx,
                accepted=accept_new,
                qwk=f"{summary['qwk']:.4f}",
                avg_misconf=f"{summary['avg_misconf']:.4f}",
                failed=metrics["failed_count"],
            )
            temperature *= alpha

        log_progress("optimization", "ASRO optimization finished", best_misconf=best_overall_misconf)
        return best_guideline

    def priority_consolidate(self, mode_rules_pool, p_current):
        if not mode_rules_pool:
            log_progress("consolidate", "no candidates; keeping current Gar")
            return p_current.get("Gar", "")

        candidates = []
        for guideline in mode_rules_pool:
            if not guideline:
                continue
            mode = guideline.get("target_mode", (0, 0))
            candidates.append(
                {
                    "mode": f"{mode[0] / 2.0} -> {mode[1] / 2.0}",
                    "content": guideline.get("Gar", ""),
                }
            )

        user_prompt = """
Merge the following patches into the current Gar.

[CURRENT GAR]
{current_gar}

[NEW PATCHES TO INTEGRATE]
{patches}

[MISSION]
Synthesize them into a single, cohesive Markdown section. Resolve contradictions.
Output # MASTER ADAPTATION RULE (GAR) only.
""".format(
            current_gar=p_current.get("Gar", "None"),
            patches="\n".join(f"- For error mode {c['mode']}: {c['content']}" for c in candidates),
        )

        log_progress("consolidate", "LLM merge request started", candidates=len(candidates))
        final_res = self._call_llm_compat(
            "You are a Senior Rubric Architect.",
            user_prompt,
            is_reflector=True,
        )
        if final_res:
            log_progress("consolidate", "LLM merge request finished", response_chars=len(final_res))
            return _legacy_cleanup(final_res)
        log_progress("consolidate", "LLM merge returned empty response")
        return p_current.get("Gar", "")

    def _get_top_k_modes(self, results, k):
        matrix_size = int(round(self.max_score * 2)) + 1
        weighted_cm = np.zeros((matrix_size, matrix_size))
        for result in results:
            t_idx = int(round(float(result["true"]) * 2))
            p_idx = int(round(float(result["pred"]) * 2))
            if t_idx != p_idx and 0 <= t_idx < matrix_size and 0 <= p_idx < matrix_size:
                weighted_cm[t_idx][p_idx] += float(result["misconf"])
        top_indices = np.argsort(weighted_cm.flatten())[::-1][:k]
        return [
            (int(divmod(int(idx), matrix_size)[0]), int(divmod(int(idx), matrix_size)[1]))
            for idx in top_indices
            if weighted_cm.flatten()[idx] > 0
        ]

    def _is_mode(self, result, mode):
        return int(round(float(result["true"]) * 2)) == mode[0] and int(round(float(result["pred"]) * 2)) == mode[1]

    def _get_contrastive_examples(self, results, mode, n=4):
        true_idx, pred_idx = mode
        same_true = [
            r
            for r in results
            if int(round(float(r["true"]) * 2)) == true_idx and int(round(float(r["pred"]) * 2)) == true_idx
        ]
        same_pred = [
            r
            for r in results
            if int(round(float(r["true"]) * 2)) == pred_idx and int(round(float(r["pred"]) * 2)) == pred_idx
        ]
        return sorted(same_true, key=lambda x: x["misconf"])[:n], sorted(same_pred, key=lambda x: x["misconf"])[:n]

    def _generate_cm_report(self, results):
        stats = {}
        for result in results:
            if int(round(float(result["true"]) * 2)) != int(round(float(result["pred"]) * 2)):
                key = f"{result['true']}->{result['pred']}"
                stats[key] = stats.get(key, 0) + 1
        sorted_stats = sorted(stats.items(), key=lambda item: item[1], reverse=True)[:5]
        return ", ".join([f"{key} ({count}pcs)" for key, count in sorted_stats])
