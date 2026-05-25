import json
import re

from utils_asro.progress import log_progress


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

    def reflector_step(self, g_k, mode_errors, contrastive_data, mode_pair, global_cm_str, curr_round=1, **payload):
        t_score = mode_pair[0] / 2.0
        p_score = mode_pair[1] / 2.0

        error_details = self._format_ASRO_examples(mode_errors[:3], "Error")
        c_true_str = self._format_ASRO_examples(contrastive_data["target_true_examples"][:2], "Correct_True")
        c_pred_str = self._format_ASRO_examples(contrastive_data["target_pred_examples"][:2], "Correct_Pred")

        from prompts import REFLECTOR_SYSTEM_PROMPT

        prompt = REFLECTOR_SYSTEM_PROMPT.format(
            true_score=t_score,
            pred_score=p_score,
            global_cm_str=global_cm_str,
            current_rubric=g_k.get("Gar", "No guideline provided"),
            error_examples_str=error_details,
            correct_true_examples_str=c_true_str,
            correct_pred_examples_str=c_pred_str,
        )

        log_progress("reflector", "prompt prepared", round=curr_round, mode=f"{t_score}->{p_score}", error_examples=len(mode_errors))

        try:
            raw_response, parsed_response = self._call_json_with_repair(
                "You are an expert English Language Assessment Specialist.",
                prompt,
                mode_pair,
                "reflector",
                curr_round,
                **payload,
            )
            log_progress("reflector", "JSON diagnosis parsed", round=curr_round, mode=f"{t_score}->{p_score}")
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
        other_modes_str = ", ".join([f"{m[0] / 2.0}->{m[1] / 2.0}" for m in other_modes_list])

        from prompts import REFINER_SYSTEM_PROMPT

        user_prompt = REFINER_SYSTEM_PROMPT.format(
            current_rubric=g_k.get("Gar", ""),
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
                return refiner_data["full_refined_rubric"]

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

    def _call_json_with_repair(
        self,
        system_prompt,
        user_prompt,
        mode_pair,
        stage_name,
        curr_round,
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
                    "JSON parse failed; retrying",
                    round=curr_round,
                    mode=f"{mode_pair[0] / 2.0}->{mode_pair[1] / 2.0}",
                    attempt=attempt + 1,
                    error=exc,
                )
                current_user_prompt = (
                    f"{user_prompt}\n\nYour previous response could not be parsed as JSON: {exc}. "
                    "Regenerate the answer as one valid JSON object only. "
                    "Do not include markdown fences, comments, or extra text."
                )

        error = OptimizerStepError(
            f"{stage_name} JSON content is malformed after repair attempts",
            stage=stage_name,
            mode_pair=mode_pair,
            raw_response=raw_response,
        )
        raise error

    def _format_ASRO_examples(self, examples, label, max_len=400):
        rendered = ""
        for idx, example in enumerate(examples):
            text = example.get("text", "")[:max_len]
            true_score = example.get("true", example.get("true_score", "N/A"))
            pred_score = example.get("pred", "N/A")
            reason = example.get("reasoning", "No reasoning provided")

            rendered += f"[{label} Case {idx + 1}]\n"
            rendered += f"Content: {text}...\n"
            rendered += f"Target: {true_score} | AI Result: {pred_score}\n"
            rendered += f"AI Reasoning: {reason}\n"
            rendered += "-" * 20 + "\n"
        return rendered

    def _legacy_cleanup(self, text):
        text = re.sub(r"```[a-zA-Z]*\n?", "", text)
        text = text.replace("```", "").strip()
        for prefix in ["Adaptation Rules:", "Final Rubric:", "Refined Rubric:"]:
            if prefix in text:
                text = text.split(prefix)[-1].strip()
        return text
