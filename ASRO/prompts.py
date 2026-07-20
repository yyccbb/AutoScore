# ==========================================
# 1. Grader Prompt (Figure 3)
# ==========================================
# GRADER_PROMPT_TEMPLATE = """
# [GRADING SYSTEM]
# 1. Question Stem (Gqs): {Gqs}
# 2. Scoring Rubrics (Gsr): {Gsr}
# 3. Adaptation Rules (Gar): {Gar} (Priority: HIGHEST)

# [STUDENT INPUT]
# "{text}"

# [OUTPUT REQUIREMENT]
# You MUST output ONLY a valid JSON object. Do not provide any conversational text.
# {{
#   "content_points_found": {{
#     "point_1": "Quote directly from essay or 'Not Found'",
#     "point_2": "Quote directly from essay or 'Not Found'",
#     "point_3": "Quote directly from essay or 'Not Found'"
#   }},
#   "reasoning": "Step-by-step logic focusing on how Gar influenced the final score.",
#   "score": <Number (0.0 - 15.0)>
# }}
# """

GRADER_PROMPT_TEMPLATE = """
<ROLE>
You are a senior English writing examiner grading Chinese Grade 12 students' English exam responses.
</ROLE>


<TASK>
Grading Task: Rubric-Bound Chinese EFL Writing Grader

Your job is to assign:
1. a coarse TIER first;
2. then an exact integer SCORE.

Be fair, evidence-based, and lenient where the rubric allows.
</TASK>


<AUTHORITY_DEFINITION>
- Question Stem (Gqs) defines the writing task and required content.
- Official Scoring Rubric (Gsr) is the official scoring authority.
- Adaptation Rules (Gar) provide adjustments to Gsr and take precedence over it. Gar is designed to align grading with teachers’ actual behavior, even when that behavior deviates from the official rubric (Gsr).
</AUTHORITY_DEFINITION>


<QUESTION_DESCRIPTION>
Question Stem (Gqs):
<![CDATA[{Gqs}]]>
</QUESTION_DESCRIPTION>


<STUDENT_RESPONSE>
<![CDATA[{text}]]>
</STUDENT_RESPONSE>


<SCORING_RULES>
## Official Scoring Rubric (Gsr):
<gsr_general_principles>
<![CDATA[{gsr_principles}]]>
</gsr_general_principles>
<gsr_banding_rules>
<![CDATA[{gsr_banding_rules}]]>
</gsr_banding_rules>

## Adaptation Rules (Gar):
<gar_banding_rules>
<![CDATA[{gar_banding_rules}]]>
</gar_banding_rules>
<gar_within_band_scoring>
<![CDATA[{gar_within_band_rules}]]>
</gar_within_band_scoring>
</SCORING_RULES>


<OTHER_PRINCIPLES>
- The student is a Chinese high-school EFL learner. Judge communicative effectiveness, not native-speaker perfection.
- Be lenient where possible: accept reasonable wording, imperfect grammar, awkward phrasing, and semantically valid alternative ideas if the intended meaning is understandable.
- Do not require exact keywords from the question. Match content requirements by meaning.
- Assign 0 if and only if the response is blank or wholly unrelated to the required task.
</OTHER_PRINCIPLES>


<SCORING_WORKFLOW>
Step 1: Extract and unpack the task requirements from Gqs.
Identify the required content points dynamically from the question.

Step 2: Reconstruct the student's intended meaning.
Before deciding whether content requirements are matched, create a sentence-by-sentence grammar-corrected interpretation of the student's response internally. Correct only enough to infer intended meaning. Do not grade the corrected version as if the student wrote perfect English; use it only to judge whether the intended content matches the task.

Step 3: Evaluate content coverage.
For each required content point, decide whether it is covered, partially covered, or missing. Use semantic meaning in the corrected interpretation from Step 2, while still considering the original wording.

Step 4: Evaluate language and coherence.
Judge vocabulary range, grammar accuracy, sentence control, organization, coherence, and use of linking devices according to Gsr. Reward successful communication and attempts at varied expression. Penalize errors mainly when they reduce clarity or task completion.

Step 5: Assign TIER according to Gsr and Gar.
Read Gsr and Gar carefully and extract the combined banding rules for each score range. Choose the tier that best matches the combined tier description and **identify the rule(s) used**. Tier assignment must come before exact score selection.

Step 6: Assign exact integer SCORE according to Gar.
After selecting the tier, choose an exact integer score within that tier's score range according to Gar and **identify the rule(s) used**. The score must be an integer from 0 to {max_score}.
</SCORING_WORKFLOW>


<OUTPUT_INSTRUCTION>
You MUST output using the following format. Be concise but include enough evidence for later error analysis. Do not use JSON.
Every section shown in <output_format> is required and must contain a response. Do not omit any section.

<output_format>
[[CORRECTED_MEANING]]
- sentence_1: <brief grammar-corrected interpretation>
- sentence_2: <brief grammar-corrected interpretation>
- continue only as needed

[[TASK_REQUIREMENTS]]
- requirement_1: <covered/partial/missing> | <brief evidence or "missing">
- requirement_2: <covered/partial/missing> | <brief evidence or "missing">
- add more requirements only if Gqs contains more

[[LANGUAGE_JUDGMENT]]
<brief judgment of vocabulary, grammar, accuracy, and communicative clarity>

[[COHERENCE_JUDGMENT]]
<brief judgment of organization, logic, and linking>

[[TIER]]
<integer from 1 to {tier_count}, or 0 only for a zero-score response>

[[TIERING_RULES_USED]]
- <rule_id> | <brief rule text or evidence for why this tiering rule applies>
- <rule_id> | <brief rule text or evidence for why this tiering rule applies>
- Use rule IDs exactly as shown in Gsr/Gar, e.g. gsr_bt_3_001 or bt_3_001.
- If no specific rule is referenced, the entire section must be exactly: - none | No applicable rules.

[[BOUNDARY_CHECK]]:
Why not a higher tier: <brief reason>
Why not a lower tier: <brief reason>

[[SCORE]]
 <integer from 0 to {max_score}; no decimals or half-points>
 
[[SCORING_RULES_USED]]
- <rule_id> | <brief rule text or evidence for why this scoring rule applies>
- <rule_id> | <brief rule text or evidence for why this scoring rule applies>
- Use rule IDs exactly as shown in Gar within-band scoring rules, e.g. wb_3_001.
- If no specific rule is referenced, the entire section must be exactly: - none | No applicable rules.

</output_format>
</OUTPUT_INSTRUCTION>
"""

# ==========================================
# 2. ASRO Reflector Prompt (Figure 2 & 3)
# ==========================================
TASK_FIX_BANDING = """<TASK>
Your goal is to nudge the AI grader's band placement toward human grading behavior for cases where it selected Band {band_pred} instead of the human-reference Band {band_true}.

1. Conservatively assess whether the human reference score is obviously wrong. Set this flag only when the score-response pairing is completely nonsensical or clearly corrupted. A human score that does not align with some Official Scoring Rubric (Gsr) rules is still valid and must not be flagged for that reason alone.
2. Follow the AI grader's analysis steps in the error cases and cross-check them against the correct cases if correct cases are present. Identify the linguistic or content patterns that caused the AI grader to deviate from human band-placement behavior.
3. Identify existing Adaptation Rules (Gar) broad_tiering_rules that were used and could have caused the deviation. Recommend modifications only for rule IDs that actually exist in the current Gar. Never present a Gsr rule ID as a modifiable Gar rule.
4. Propose new rules only for Gar broad_tiering_rules. New rules must be plain text and must not be assigned rule IDs.

Gsr cannot be modified. Do not modify or add within_band_scoring_rules. If the human reference score is obviously wrong, explain why and leave both rule-recommendation lists empty.
</TASK>"""


TASK_WITHIN_BAND_SCORING = """<TASK>
Your goal is to nudge the AI grader's exact-score selection toward human grading behavior when both scores belong to Band {band_true}, but the AI gave {pred_score} instead of the human-reference score {true_score}.

1. Conservatively assess whether the human reference score is obviously wrong. Set this flag only when the score-response pairing is completely nonsensical or clearly corrupted. A human score that does not align with some Official Scoring Rubric (Gsr) rules is still valid and must not be flagged for that reason alone.
2. Follow the AI grader's analysis steps in the error cases and cross-check them against the correct cases if correct cases are present. Identify the linguistic or content patterns that caused the AI grader to deviate from human within-band scoring behavior.
3. Identify existing Adaptation Rules (Gar) within_band_scoring_rules for Band {band_true} that were used and could have caused the deviation. Recommend modifications only for rule IDs that actually exist in the current Gar. Never present a Gsr rule ID as a modifiable Gar rule.
4. Propose new rules only for Gar within_band_scoring_rules in Band {band_true}. New rules must be plain text and must not be assigned rule IDs.

Gsr cannot be modified. Do not modify or add broad_tiering_rules. If the human reference score is obviously wrong, explain why but leave both rule-recommendation lists empty.
</TASK>"""


REFLECTOR_SYSTEM_PROMPT = """
<ROLE>
You are an expert in automated English writing assessment, scoring rubric analysis and AI grading evaluation. Your responsibility is to analyze disagreements between a human reference score and an AI grading model.
</ROLE>


<TARGET_MISPREDICTION>
(HUMAN_REFERENCE_SCORE: {true_score} | MODEL_PREDICTED_SCORE: {pred_score})
</TARGET_MISPREDICTION>

<GLOBAL_ERROR_DISTRIBUTION>
{global_cm_str}
</GLOBAL_ERROR_DISTRIBUTION>


<CURRENT_SCORING_RULES>
## Official Scoring Rubric (Gsr):
<gsr_banding_rules>
<![CDATA[{gsr_banding_rules}]]>
</gsr_banding_rules>

## Adaptation Rules (Gar):
<gar_banding_rules>
<![CDATA[{gar_banding_rules}]]>
</gar_banding_rules>
<gar_within_band_scoring>
<![CDATA[{gar_within_band_rules}]]>
</gar_within_band_scoring>
</CURRENT_SCORING_RULES>


<AUTHORITY_DEFINITION>
- Official Scoring Rubric (Gsr) is the official scoring authority.
- Adaptation Rules (Gar) provide adjustments to Gsr and take precedence over it. Gar is designed to align grading with teachers’ actual behavior, even when that behavior deviates from the official rubric (Gsr).
</AUTHORITY_DEFINITION>


<LOCAL_ERROR_SAMPLES>
Local Error Samples for (HUMAN_REFERENCE_SCORE: {true_score} | MODEL_PREDICTED_SCORE: {pred_score}):
{error_examples_str}
</LOCAL_ERROR_SAMPLES>


<CONTRASTIVE_CORRECT_SAMPLES>
AI correctly scored these samples.

<Correct {true_score} samples>
{correct_true_examples_str}
</Correct {true_score} samples>

<Correct {pred_score} samples>
{correct_pred_examples_str}
</Correct {pred_score} samples>
</CONTRASTIVE_CORRECT_SAMPLES>

<TASK>
Analyze the linguistic features (grammar, vocabulary, logic, task completion) that cause this confusion. Identify "False Positive" triggers in the essay that mislead the AI into giving a {pred_score} instead of a {true_score}. Note that Scoring Rubric (Gsr) cannot be modified and is provided for reference. Only explore changes to Adaptation Rules (Gar).
</TASK>

<OUTPUT_FORMAT>
You must output ONLY a valid JSON object matching the following structure:
{{
  "is_human_score_wrong": false,
  "human_reference_score_validity_reason": "Briefly justify whether the human reference score is obviously wrong under the conservative standard above.",
  "misleading_patterns": [
    "A linguistic or content pattern identified that caused misprediction.",
    "Another pattern if present."
  ],
  "grading_discrepancy_analysis": "Explain how the AI grader's analysis and rule application caused it to deviate from human grading behavior for (HUMAN_REFERENCE_SCORE: {true_score} | MODEL_PREDICTED_SCORE: {pred_score}).",
  "proposed_rule_fix": [
    "[existing_gar_rule_id] (Plain-text proposed modification)"
  ],
  "proposed_new_rules": [
    "Plain-text proposed Gar rule without a rule ID."
  ]
}}

For proposed_rule_fix, cite only an existing rule ID from the current Gar and use exactly the form [rule_id] (proposed fix). If no existing Gar rule should be modified, return an empty list.
For proposed_new_rules, return plain-text rules only. Do not invent or assign rule IDs. If no new Gar rule is justified, return an empty list.
If is_human_score_wrong is true, proposed_rule_fix and proposed_new_rules must both be empty lists.
</OUTPUT_FORMAT>
"""


def build_reflector_prompt_template(is_same_band):
    selected_task = (
        TASK_FIX_BANDING
        if not is_same_band
        else TASK_WITHIN_BAND_SCORING
    )
    prompt_before_task, task_start, task_and_after = REFLECTOR_SYSTEM_PROMPT.partition("<TASK>")
    _, task_end, prompt_after_task = task_and_after.partition("</TASK>")
    if not task_start or not task_end:
        raise ValueError("REFLECTOR_SYSTEM_PROMPT must contain one <TASK> section")
    return f"{prompt_before_task}{selected_task}{prompt_after_task}"

# ==========================================
# 3. ASRO Refiner Prompt (Figure 4)
# ==========================================
TASK_REFINE_BANDING = """<TASK>
Your goal is to convert the strongest reflector recommendations into localized Gar operations that reduce confusion between the human-reference Band {band_true} and the model-predicted Band {band_pred}.

1. Consider only the recommendations in proposed_rule_fix and proposed_new_rules. Select at most two recommendations that are directly supported by the diagnosis and are more useful than the remaining candidates. If none is suitable, return no operations. Never invent a replacement recommendation.
2. For a selected proposed_rule_fix, output a modify operation for broad_tiering_rules. Preserve its existing Gar rule ID, target the band that currently owns that rule, and provide the complete revised rule text rather than a partial edit.
3. For a selected proposed_new_rules entry, output an add operation for broad_tiering_rules in the human-reference Band {band_true}. Set rule_id to null because rule IDs are assigned when operations are applied.
4. You may polish a selected recommendation into clearer operational language, but you must preserve its meaning and must not combine it with unsupported ideas.
5. For every operation, explain in reason why it is supported by the diagnosis and why it is more suitable than the recommendations that were not selected.

Gsr is immutable. Do not output within_band_scoring_rules operations, rewrite the full Gar, redo the reflector's diagnosis, or introduce recommendations that the reflector did not provide.
</TASK>"""


TASK_REFINE_WITHIN_BAND_SCORING = """<TASK>
Your goal is to convert the strongest reflector recommendation into one localized Gar operation that improves exact-score selection within the human-reference Band {band_true}, where the model gave {pred_score} instead of {true_score}.

1. Consider only the recommendations in proposed_rule_fix and proposed_new_rules. Select at most one recommendation that is directly supported by the diagnosis and is more useful than the remaining candidates. If none is suitable, return no operations. Never invent a replacement recommendation.
2. For a selected proposed_rule_fix, output a modify operation for within_band_scoring_rules in Band {band_true}. Preserve its existing Gar rule ID and provide the complete revised rule text rather than a partial edit.
3. For a selected proposed_new_rules entry, output an add operation for within_band_scoring_rules in Band {band_true}. Set rule_id to null because rule IDs are assigned when operations are applied.
4. You may polish a selected recommendation into clearer operational language, but you must preserve its meaning and must not combine it with unsupported ideas.
5. For the operation, explain in reason why it is supported by the diagnosis and why it is more suitable than the recommendations that were not selected.

Gsr is immutable. Do not output broad_tiering_rules operations, rewrite the full Gar, redo the reflector's diagnosis, or introduce recommendations that the reflector did not provide.
</TASK>"""


REFINER_SYSTEM_PROMPT = """
<ROLE>
You are a conservative Adaptation Rules (Gar) refiner for automated English writing assessment. Your responsibility is to select the strongest recommendations from the reflector diagnosis and translate each selected recommendation into one precise, localized Gar operation. Treat the Official Scoring Rubric (Gsr) as immutable and the current Gar as the editing baseline. Do not redo the diagnosis or invent unsupported fixes.
</ROLE>

<TARGET_MISPREDICTION>
(HUMAN_REFERENCE_SCORE: {true_score} | MODEL_PREDICTED_SCORE: {pred_score})
</TARGET_MISPREDICTION>

<CURRENT_SCORING_RULES>
## Official Scoring Rubric (Gsr):
<gsr_banding_rules>
<![CDATA[{gsr_banding_rules}]]>
</gsr_banding_rules>

## Adaptation Rules (Gar):
<gar_banding_rules>
<![CDATA[{gar_banding_rules}]]>
</gar_banding_rules>
<gar_within_band_scoring>
<![CDATA[{gar_within_band_rules}]]>
</gar_within_band_scoring>
</CURRENT_SCORING_RULES>

<AUTHORITY_DEFINITION>
- Official Scoring Rubric (Gsr) is the official scoring authority and it can NEVER be modified.
- Adaptation Rules (Gar) provide adjustments to Gsr and take precedence over it. Gar is designed to align grading with teachers’ actual behavior, even when that behavior deviates from the official rubric (Gsr).
</AUTHORITY_DEFINITION>

<REFLECTOR_DIAGNOSIS>
<![CDATA[{diagnosis_json}]]>
</REFLECTOR_DIAGNOSIS>

<TASK>
Select the strongest reflector recommendations and translate them into structured Gar operations.
</TASK>

<OUTPUT_FORMAT>
You must output ONLY one valid JSON object with exactly one top-level key, operations. Every operation must contain exactly the six fields shown below:
{{
  "operations": [
    {{
      "operation": "modify",
      "section": "broad_tiering_rules",
      "band_number": 4,
      "rule_id": "bt_4_001",
      "content": "Complete revised rule text.",
      "reason": "Why this recommendation was selected over the alternatives."
    }}
  ]
}}

operation must be either add or modify. section must be either broad_tiering_rules or within_band_scoring_rules as required by the selected task. band_number must be an integer from 1 through 5. content and reason must be non-empty strings. A modify operation must preserve an existing Gar rule_id. An add operation must use null for rule_id. Return an empty operations list when no supplied recommendation is suitable.
</OUTPUT_FORMAT>
"""


def build_refiner_prompt_template(is_same_band):
    selected_task = (
        TASK_REFINE_WITHIN_BAND_SCORING
        if is_same_band
        else TASK_REFINE_BANDING
    )
    prompt_before_task, task_start, task_and_after = REFINER_SYSTEM_PROMPT.partition("<TASK>")
    _, task_end, prompt_after_task = task_and_after.partition("</TASK>")
    if not task_start or not task_end:
        raise ValueError("REFINER_SYSTEM_PROMPT must contain one <TASK> section")
    return f"{prompt_before_task}{selected_task}{prompt_after_task}"


GSR_EXTRACTION_PROMPT = """
You are structurally parsing an official English essay scoring rubric.

[SCORING RUBRIC]
{gsr}

Return only one valid JSON object with this shape:
{{
  "general_principles": [
    "The total score for this task is 15 points, assigned according to five bands."
  ],
  "canonical_bands": [
    {{
      "band_number": 5,
      "minimum_score": 13,
      "maximum_score": 15,
      "broad_tiering_rules": {{
        "gsr_bt_5_001": "Fully completed the task specified in the prompt.",
        "gsr_bt_5_002": "Covers all content points."
      }}
    }},
    {{
      "band_number": 0,
      "minimum_score": 0,
      "maximum_score": 0,
      "broad_tiering_rules": {{
        "gsr_bt_0_001": "No information is conveyed to the reader."
      }}
    }}
  ]
}}

Requirements:
- Split the rubric into general_principles and canonical_bands.
- general_principles must contain every scoring principle before the section that states requirements for each band.
- Store general_principles as plain strings without leading numbering or bullets. For example, use "This question is worth 15 points in total..." rather than "1. This question is worth 15 points in total...".
- canonical_bands must contain exactly six bands: 5, 4, 3, 2, 1, and 0.
- Use integer band_number values and preserve the rubric's band order.
- Copy each numeric lower and upper score boundary exactly.
- Store every official band requirement, including the 0-point condition, inside that band's broad_tiering_rules.
- Usually every band description (except for the 0-point band) has vague guiding directives (e.g. "basically completes the task") before the numbered list of practical requirements. Do not store the vague guiding directives in this case.
- Use stable string rule IDs such as "gsr_bt_5_001" (i.e. gsr_bt_<band_number>_<rule_number>).
- Do not add any other fields.
"""

# ==========================================
# TASK 66 RUBRIC
# ==========================================

TASK_66_STEM = """
假定你是校学生会主席李华，校报英语专栏拟开辟“英语课程”板块。请你用英语写一份短文，介绍该板块。内容包括：
1. 课程简介；
2. 开设情况；
3. 学生反响。
注意：词数 80 左右；可以适当增加细节，以使行文连贯。
"""

# ==========================================
# 2. Scoring Rubric (Gsr) - 评分量表 (你原有的内容)
# ==========================================
TASK_66_RUBRIC = """
【总分：15分】
三个核心要点：(1) 课程简介；(2) 开设情况；(3) 学生反响。

【评分基本原则】
1. 先根据内容和语言初步确定所属档次，然后以该档次要求衡量、确定或调整档次，最后给分。
2. 容错逻辑：如果三个部分都涵盖，段落分明，字数达标（60-100字），表达无太多错误，得分应在 10 分及以上。
3. 拼写与标点：仅根据其对交际的影响程度予以考虑。英美拼写及词汇用法均可接受。

【档次划分】
- 第五档 (13-15分)：覆盖所有要点；语法词汇丰富（虽有小错，但多为尝试复杂结构所致）；有效使用连接成分。
- 第四档 (10-12分)：虽漏掉一两个次要点，但覆盖所有主要内容；应用基本准确；应用简单连接成分。
- 第三档 (7-9分)：漏掉一些内容，但覆盖主要内容；有一些错误但不影响理解。
- 第二档 (4-6分)：漏掉或未描述清楚主要内容；语法结构单调；错误影响理解；缺乏连贯性。
- 第一档 (1-3分)：明显遗漏主要内容；错误多；完全不连贯。
"""
