import json
import re

from utils_asro.gar_models import GAR, gar_from_value, gar_to_json, gsr_from_value, gsr_to_text
from utils_asro.progress import log_progress

COMMON_REFLECTOR_ANALYSIS_TAGS = (
    "TASK_REQUIREMENTS",
    "LANGUAGE_JUDGMENT",
    "COHERENCE_JUDGMENT",
)

REFLECTOR_DIAGNOSIS_KEYS = {
    "is_human_score_wrong",
    "human_reference_score_validity_reason",
    "misleading_patterns",
    "grading_discrepancy_analysis",
    "proposed_rule_fix",
    "proposed_new_rules",
}


class OptimizerStepError(RuntimeError):
    def __init__(self, message, stage=None, mode_pair=None, raw_response=None):
        super().__init__(message)
        self.stage = stage
        self.mode_pair = mode_pair
        self.raw_response = raw_response


class GradeOptimizer:
    def __init__(self, client, output_dir="."):
        self.client = client
        self.output_dir = output_dir

    def _call_llm_compat(self, system_prompt, user_prompt, **payload):
        call_llm = getattr(self.client, "call_llm")
        code = getattr(call_llm, "__code__", None)
        varnames = code.co_varnames[: code.co_argcount] if code else ()
        if len(varnames) >= 3 and varnames[1:3] == ("prompt", "is_reflector"):
            prompt = f"{system_prompt}\n\n{user_prompt}".strip()
            return call_llm(prompt, is_reflector=payload.get("is_reflector", True))
        payload = {k: v for k, v in payload.items() if k != "is_reflector"}
        return call_llm(system_prompt, user_prompt, **payload)

    def _format_reflector_rubric_context(self, guideline):
        def normalize(value):
            if value is None:
                return ""
            if isinstance(value, str):
                return value.strip()
            try:
                return json.dumps(value, indent=2, ensure_ascii=False).strip()
            except TypeError:
                return str(value).strip()

        guideline = guideline or {}
        gsr = gsr_to_text(guideline.get("Gsr"))
        raw_gar = guideline.get("Gar")
        gar = gar_to_json(raw_gar) if raw_gar else ""
        sections = []

        if gsr:
            sections.append(f"## Scoring Rubric (Gsr)\n{gsr}")
        if gar:
            sections.append(f"## Adaptation Rules (Gar)\n{gar}")

        return "\n\n".join(sections) or "No guideline provided"

    def reflector_step(self, p_current, mode_errors, contrastive_data, mode_pair, global_cm_str, curr_round=1, **payload):
        score_true = mode_pair[0] / 2.0
        score_pred = mode_pair[1] / 2.0
        structured_gsr = gsr_from_value(p_current["Gsr"])
        structured_gar = gar_from_value(p_current["Gar"])
        band_true = next(
            band.band_number
            for band in structured_gsr.canonical_bands
            if band.minimum_score <= score_true <= band.maximum_score
        )
        band_pred = next(
            band.band_number
            for band in structured_gsr.canonical_bands
            if band.minimum_score <= score_pred <= band.maximum_score
        )

        error_details = self._format_ASRO_examples(
            mode_errors[:3],
            "Error",
            band_true,
            band_pred,
        )
        c_true_str = self._format_ASRO_examples(
            contrastive_data["target_true_examples"][:2],
            "Correct",
            band_true,
            band_true,
        )
        c_pred_str = self._format_ASRO_examples(
            contrastive_data["target_pred_examples"][:2],
            "Correct",
            band_pred,
            band_pred,
        )
        if not c_true_str.strip():
            c_true_str = "No true examples."
        if not c_pred_str.strip():
            c_pred_str = "No predicted examples."

        from prompts import build_reflector_prompt_template

        is_same_band = band_true == band_pred
        if is_same_band:
            allowed_rule_ids = {
                rule_id
                for band in structured_gar.canonical_bands
                if band.band_number == band_true
                for rule_id in band.within_band_scoring_rules
            }
        else:
            allowed_rule_ids = {
                rule_id
                for band in structured_gar.canonical_bands
                for rule_id in band.broad_tiering_rules
            }

        prompt_template = build_reflector_prompt_template(is_same_band)
        prompt = prompt_template.format(
            true_score=score_true,
            pred_score=score_pred,
            band_true=band_true,
            band_pred=band_pred,
            global_cm_str=global_cm_str,
            gsr_banding_rules=structured_gsr.banding_rules_text(),
            gar_banding_rules=structured_gar.banding_rules_text(),
            gar_within_band_rules=structured_gar.within_band_scoring_rules_text(),
            error_examples_str=error_details,
            correct_true_examples_str=c_true_str,
            correct_pred_examples_str=c_pred_str,
        )

        log_progress("reflector", "prompt prepared", round=curr_round, mode=f"{score_true}->{score_pred}", error_examples=len(mode_errors))

        try:
            raw_response, parsed_response = self._call_json_with_repair(
                "",
                prompt,
                mode_pair,
                "reflector",
                curr_round,
                response_validator=lambda response: self._validate_reflector_diagnosis(
                    response,
                    allowed_rule_ids,
                ),
                **payload,
            )
            log_progress("reflector", "JSON diagnosis parsed", round=curr_round, mode=f"{score_true}->{score_pred}")
            return parsed_response
        except Exception as exc:
            raise OptimizerStepError(
                f"Reflector failed for mode {mode_pair}: {exc}",
                stage="reflector",
                mode_pair=mode_pair,
                raw_response=getattr(exc, "raw_response", None),
            ) from exc

    def refiner_step(self, g_k, diagnosis_json, other_modes_list, mode_pair, curr_round=1, **payload):
        t_score = mode_pair[0] / 2.0
        p_score = mode_pair[1] / 2.0
        other_modes_str = ", ".join(
            [
                f"(HUMAN_REFERENCE_SCORE: {m[0] / 2.0} | "
                f"MODEL_PREDICTED_SCORE: {m[1] / 2.0})"
                for m in other_modes_list
            ]
        )

        from prompts import REFINER_SYSTEM_PROMPT

        user_prompt = REFINER_SYSTEM_PROMPT.format(
            current_rubric=gar_to_json(g_k["Gar"]),
            diagnosis_json=json.dumps(diagnosis_json, indent=2, ensure_ascii=False),
            error_examples_str="[See provided analysis]",
            other_modes_context=f"Warning: Conflicts may arise with: {other_modes_str}",
            true_score=t_score,
            pred_score=p_score,
            other_modes_str=other_modes_str if other_modes_str else "None",
        )

        log_progress("refiner", "prompt prepared", round=curr_round, mode=f"{t_score}->{p_score}")

        try:
            raw_response, refiner_data = self._call_json_with_repair(
                "You are a Senior Rubric Architect.",
                user_prompt,
                mode_pair,
                "refiner",
                curr_round,
                **payload,
            )
            log_progress("refiner", "JSON patch parsed", round=curr_round, mode=f"{t_score}->{p_score}")

            if isinstance(refiner_data, dict) and "full_refined_rubric" in refiner_data:
                return GAR.from_dict(refiner_data["full_refined_rubric"])

            raise ValueError("Refiner JSON does not contain full_refined_rubric")
        except Exception as exc:
            raise OptimizerStepError(
                f"Refiner failed for mode {mode_pair}: {exc}",
                stage="refiner",
                mode_pair=mode_pair,
                raw_response=getattr(exc, "raw_response", None),
            ) from exc

    def _strip_fences(self, text):
        text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
        fenced = re.search(r"```json\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            return fenced.group(1).strip()
        text = re.sub(r"```[a-zA-Z]*\s*", "", text)
        text = text.replace("```", "")
        return text.strip()

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

    def _extract_json(self, text):
        cleaned = self._strip_fences(text)
        json_text = self._first_json_object(cleaned)
        try:
            return json.loads(json_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON content is malformed: {exc}") from exc

    def _validate_reflector_diagnosis(self, response, allowed_rule_ids):
        if not isinstance(response, dict):
            raise ValueError("Reflector response must be a JSON object")

        response_keys = set(response)
        missing_keys = sorted(REFLECTOR_DIAGNOSIS_KEYS - response_keys)
        unexpected_keys = sorted(response_keys - REFLECTOR_DIAGNOSIS_KEYS)
        if missing_keys or unexpected_keys:
            details = []
            if missing_keys:
                details.append(f"missing keys: {', '.join(missing_keys)}")
            if unexpected_keys:
                details.append(f"unexpected keys: {', '.join(unexpected_keys)}")
            raise ValueError("Reflector response has " + "; ".join(details))

        if type(response["is_human_score_wrong"]) is not bool:
            raise ValueError("is_human_score_wrong must be a JSON boolean")

        for field_name in (
            "human_reference_score_validity_reason",
            "grading_discrepancy_analysis",
        ):
            value = response[field_name]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")

        for field_name in (
            "misleading_patterns",
            "proposed_rule_fix",
            "proposed_new_rules",
        ):
            value = response[field_name]
            if not isinstance(value, list):
                raise ValueError(f"{field_name} must be a list")
            if any(not isinstance(item, str) or not item.strip() for item in value):
                raise ValueError(
                    f"{field_name} must contain only non-empty strings"
                )

        for proposed_fix in response["proposed_rule_fix"]:
            match = re.fullmatch(
                r"\[([^\[\]]+)\]\s+\((.+)\)",
                proposed_fix.strip(),
                flags=re.DOTALL,
            )
            if not match or not match.group(2).strip():
                raise ValueError(
                    "Each proposed_rule_fix entry must use "
                    "[rule_id] (non-empty proposed fix)"
                )
            rule_id = match.group(1)
            if rule_id != rule_id.strip() or rule_id not in allowed_rule_ids:
                raise ValueError(
                    f"proposed_rule_fix references an unavailable Gar rule ID: "
                    f"{rule_id!r}"
                )

        for proposed_rule in response["proposed_new_rules"]:
            if re.match(r"^\s*\[[^\]]+\]", proposed_rule):
                raise ValueError(
                    "proposed_new_rules entries must not be assigned rule IDs"
                )

        if response["is_human_score_wrong"] and (
            response["proposed_rule_fix"] or response["proposed_new_rules"]
        ):
            raise ValueError(
                "proposed_rule_fix and proposed_new_rules must both be empty "
                "when is_human_score_wrong is true"
            )

    def _call_json_with_repair(
        self,
        system_prompt,
        user_prompt,
        mode_pair,
        stage_name,
        curr_round,
        response_validator=None,
        max_attempts=3,
        temperature=None,
        max_tokens=None,
        timeout=None,
    ):
        current_user_prompt = user_prompt
        raw_response = ""
        payload = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
            "is_reflector": True,
        }
        payload = {k: v for k, v in payload.items() if v is not None}

        for attempt in range(max_attempts):
            log_progress(
                stage_name,
                "JSON LLM attempt started",
                round=curr_round,
                mode=f"{mode_pair[0] / 2.0}->{mode_pair[1] / 2.0}",
                attempt=attempt + 1,
                max_attempts=max_attempts,
            )
            raw_response = self._call_llm_compat(system_prompt, current_user_prompt, **payload)
            try:
                parsed = self._extract_json(raw_response)
                if response_validator is not None:
                    response_validator(parsed)
                log_progress(
                    stage_name,
                    "JSON LLM attempt parsed",
                    round=curr_round,
                    mode=f"{mode_pair[0] / 2.0}->{mode_pair[1] / 2.0}",
                    attempt=attempt + 1,
                )
                return raw_response, parsed
            except Exception as exc:
                log_progress(
                    stage_name,
                    "JSON response invalid; retrying",
                    round=curr_round,
                    mode=f"{mode_pair[0] / 2.0}->{mode_pair[1] / 2.0}",
                    attempt=attempt + 1,
                    error=exc,
                )
                current_user_prompt = (
                    f"{user_prompt}\n\nYour previous response did not satisfy the "
                    f"required JSON format: {exc}. Regenerate the answer as one "
                    "valid JSON object that follows every required key, type, and "
                    "constraint. Do not include markdown fences, comments, or extra text."
                )

        error = OptimizerStepError(
            f"{stage_name} JSON response is invalid after repair attempts",
            stage=stage_name,
            mode_pair=mode_pair,
            raw_response=raw_response,
        )
        raise error

    def _format_ASRO_examples(self, examples, label, human_tier, pred_tier):
        rendered_examples = []
        for idx, example in enumerate(examples):
            text = str(example.get("text", "")).strip()
            true_score = example.get("true", example.get("true_score", "N/A"))
            pred_score = example.get("pred", "N/A")
            sections = [
                (
                    f"[{label} Case {idx + 1}]\n"
                    f"<student_response>\n{text}\n</student_response>"
                ),
                (
                    "<scoring>\n"
                    f"Human Score: {true_score}\n"
                    f"Human Tier: {human_tier}\n"
                    f"AI Grader Score: {pred_score}\n"
                    f"AI Grader Tier: {pred_tier}\n"
                    "</scoring>"
                ),
            ]

            for tag_name in COMMON_REFLECTOR_ANALYSIS_TAGS:
                raw_value = example.get(tag_name)
                if raw_value is None:
                    continue
                value = str(raw_value).strip()
                if not value:
                    continue
                section_name = f"{tag_name.lower()}_analysis_by_ai_grader"
                sections.append(
                    f"<{section_name}>\n{value}\n</{section_name}>"
                )

            conditional_tags = (
                (
                    ("SCORING_RULES_USED", "within_tier_scoring_rules_used"),
                )
                if human_tier == pred_tier
                else (
                    ("BOUNDARY_CHECK", "boundary_check_by_ai_grader"),
                    ("TIERING_RULES_USED", "tiering_rules_used"),
                )
            )
            for tag_name, section_name in conditional_tags:
                raw_value = example.get(tag_name)
                if raw_value is None:
                    continue
                value = str(raw_value).strip()
                if value:
                    sections.append(
                        f"<{section_name}>\n{value}\n</{section_name}>"
                    )

            rendered_examples.append(
                "\n\n".join(sections) + "\n" + "-" * 20
            )

        if not rendered_examples:
            return ""
        return "\n".join(rendered_examples) + "\n"

    def _legacy_cleanup(self, text):
        text = re.sub(r"```[a-zA-Z]*\n?", "", text)
        text = text.replace("```", "").strip()
        for prefix in ["Adaptation Rules:", "Final Rubric:", "Refined Rubric:"]:
            if prefix in text:
                text = text.split(prefix)[-1].strip()
        return text
