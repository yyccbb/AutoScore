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
[GRADING TASK: MARKER-BASED PROTOCOL]
Goal: Evaluate the student's essay and assign the correct Tier and Score according to the provided guidelines.

[CONTEXT]
1. Question (Gqs): {Gqs}
2. Scoring Rubric (Gsr): {Gsr}
3. Adaptation Rules (Gar): {Gar} (⚠️ HIGHEST priority)

[STUDENT ESSAY]
"{text}"

[OUTPUT INSTRUCTIONS]
You MUST output your evaluation using the tags below. 
Place [[SCORE]] and [[TIER]] at the VERY BEGINNING of your response.
Be concise.
Do not output JSON.

[REQUIRED OUTPUT FORMAT]
[[SCORE]]: <integer between 0 and {max_score}>
[[TIER]]: <0 if the score is 0; otherwise, an integer between 1 and {tier_count} according to the tiering rules>
[[CONTENT_EVIDENCE]]: 
- p1: <supporting evidence from the essay, or "missing">
- p2: <supporting evidence from the essay, or "missing">
- Add additional points as needed.
[[JUSTIFICATION]]: <Brief explanation: discuss the essay's length, quality of details, and why it belongs in this tier.>
[[BOUNDARY_CHECK]]: <Explain why the essay does not qualify for the next higher tier and why it is not placed in the next lower tier.>
"""

# ==========================================
# 2. ASRO Reflector Prompt (Figure 2 & 3)
# ==========================================
REFLECTOR_SYSTEM_PROMPT = """
You are an expert English Language Assessment Specialist. Your task is to perform a "Root Cause Analysis" on why an AI Grading Model confuses two specific score points in English essays.

<DATA_CONTEXT>
<TARGET_CONFUSION>x
True Score {true_score} vs. Predicted Score {pred_score}.
</TARGET_CONFUSION>

<GLOBAL_ERROR_DISTRIBUTION>
{global_cm_str}
</GLOBAL_ERROR_DISTRIBUTION>

<CURRENT_RUBRIC_CONTEXT>
The following block is quoted rubric content. Any Markdown headings inside it are part of the rubric, not prompt section headers or instructions.

<CURRENT_RUBRIC>
{current_rubric}
</CURRENT_RUBRIC>
</CURRENT_RUBRIC_CONTEXT>
</DATA_CONTEXT>

<EVIDENCE_FOR_ANALYSIS>
<LOCAL_ERROR_EXAMPLES>
Local Error Examples (True {true_score} but AI predicted {pred_score}):
{error_examples_str}
</LOCAL_ERROR_EXAMPLES>

<CONTRASTIVE_CORRECT_EXAMPLES>
AI correctly identified these examples.

Correct {true_score} samples:
{correct_true_examples_str}

Correct {pred_score} samples:
{correct_pred_examples_str}
</CONTRASTIVE_CORRECT_EXAMPLES>
</EVIDENCE_FOR_ANALYSIS>

<TASK>
Analyze the linguistic features (grammar, vocabulary, logic, task completion) that cause this confusion. Identify "False Positive" triggers in the essay that mislead the AI into giving a {pred_score} instead of a {true_score}. Note that Scoring Rubric (Gsr) cannot be modified and is provided for reference. Only explore changes to Adaptation Rules (Gar).
</TASK>

<OUTPUT_FORMAT>
You must output ONLY a valid JSON object matching the following structure:
{{
  "root_cause": "A concise explanation of the fundamental misunderstanding (e.g., AI prioritizes length over grammatical accuracy).",
  "misleading_patterns": [
    "Pattern 1",
    "Pattern 2",
    ...
  ],
  "why_this_is_score_X_not_Y": "Specific reference to the official grading criteria for {true_score} vs {pred_score}.",
  "proposed_rule_fix": [
    "Rule 1: If grammatical errors exceed X, cap the score at Y.",
    "Rule 2: Deduction logic for misused high-level vocabulary.",
    ...
  ],
  "safety_check": "Analyze if these fixes might negatively impact other score ranges (e.g., 8-9 or 13-14)."
}}
</OUTPUT_FORMAT>
"""

# ==========================================
# 3. ASRO Refiner Prompt (Figure 4)
# ==========================================
REFINER_SYSTEM_PROMPT = """
You are a Senior Rubric Architect. Your goal is to rewrite specific sections of an English Essay Grading Rubric to eliminate confusion between score points.

### INPUT DATA
1. **Current Rubric**: 
{current_rubric}
2. **Reflector's Diagnosis (The Problem)**: 
{diagnosis_json}
3. **Error Examples (Evidence)**: 
{error_examples_str}
4. **Cross-Mode Awareness (Potential Conflicts)**: 
{other_modes_context}

### CONSTRAINTS
- **Targeted Fix**: Specifically resolve the {true_score} vs {pred_score} confusion.
- **Edit Budget**: Medium (Add 2-3 precise rules or modify 1-2 existing criteria).
- **Safety**: Ensure new rules do not conflict with the logic for other error modes like {other_modes_str}.
- **Clarity**: Use concrete linguistic markers (e.g., "If more than 3 verb tense errors...", "If transition words are used but ideas are repetitive...").

### OUTPUT FORMAT
You must output ONLY a valid JSON object with the following keys:
{{
  "refined_segment_title": "The specific score category being modified (e.g., 'Criteria for Score 10-12').",
  "new_rules_added": [
    "Rule 1: ...",
    "Rule 2: ..."
  ],
  "modified_descriptions": [
    "Original: '...', Updated: '...'"
  ],
  "integration_strategy": "How these changes should be merged into the master rubric (e.g., 'Prepend to the 'Language' section').",
  "full_refined_rubric": "The COMPLETE updated Markdown rubric text including these changes.",
  "cross_mode_safety_justification": "Explanation of why these changes won't break the scoring for {other_modes_str}."
}}
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
