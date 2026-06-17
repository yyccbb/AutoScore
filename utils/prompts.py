BASIC_OCR_PROMPT = """
Task: High-fidelity English Essay Transcription with Smart Noise Reduction.

CRITICAL INSTRUCTIONS:
1. CONTEXTUAL REPAIR: If a word is visually obscured or contains obvious OCR artifacts (e.g., "v@ry" instead of "very", "th1s" instead of "this"), use linguistic context to restore the INTENDED word. 
2. PRESERVE LINGUISTIC ERRORS: Do not perform "Grammatical Normalization". If the student intentionally wrote a grammatically incorrect phrase (e.g., "He go to school", "I am very happyer"), you MUST transcribe it exactly as written. Distinguish between 'digitization noise' and 'student error'.
3. HANDLE CANCELLATIONS: If the student has crossed out text with lines or scribbles, omit those parts entirely. Do not guess what was under the strike-through.
4. STRIP NON-ESSENTIALS: Ignore all Chinese characters, page headers, question numbers, and printed instructions. 
5. CONTINUATION WRITING: If the image includes printed sentence starters, transcribe them first, then seamlessly continue with the student's handwritten response.
6. STRUCTURE: Maintain original paragraphing. No headers, no commentary, just the raw English text.

Output: Only the transcribed English text.
"""

EXTRACTION_USER_TEMPLATE = """
### Required Content Points
{points}

### Student Response
{essay_text}

---
Extract components into the following JSON structure. Be extremely concise.

{{
    "content_points": [
        // Generate one object for EVERY point listed in Required Content Points
        {{
            "id": "<point_id_from_required>",
            "covered": true/false,
            "evidence": "brief quote",
            "detail": "none/brief/detailed"
        }}
    ],
    
    "language_highlights": ["list of native-like phrases or complex clauses, max 3"],
    
    "language_issues": {{
        "grammar_spelling": ["list of objective errors, max 5"],
        "chinglish_templates": ["unnatural memorized phrases, e.g., 'every coin has two sides'"],
        "repetitive_words": ["list of overused basic words"]
    }},
    
    "discourse": {{
        "logical_jumps": ["list of unclear pronouns or logic gaps, max 2"],
        "connectives": "poor/adequate/good"
    }},
    
    "task_constraints": {{
        "meets_length": true/false,
        "format_errors": ["e.g., missing salutation, or empty list if none"]
    }}
}}
"""

SCORING_USER_TEMPLATE = """
### Task Context (X)
{task_rubric}

### ASSIGNED TIER
{tier}

### Structured Representation (Z)
{evidence_z}

### Original Response (R)
{essay_text}

---
Based on the extracted representation (Z) and the task context (X), assign the final score. 
Specifically evaluate the "idiomatic_expressions" in Z: are they truly native-like, or just common EFL patterns? 
[SYSTEM NOTE]: The following student response has been processed via an OCR (Optical Character Recognition) system. Be advised that the text may contain digitization noise, character misidentifications, or formatting artifacts inherent to the scanning and extraction process. You MUST prioritize the student's communicative intent and overall linguistic proficiency. Do not penalize the score for localized character-level inconsistencies that could reasonably be attributed to digitization noise rather than the student's actual writing.
[CRITICAL] Strictly adhere to the assigned TIER as a boundary condition. If the evidence in Z does not meet the criteria for the assigned TIER, you MUST adjust the score downward accordingly. Do NOT assign a score that exceeds the upper limit of the assigned TIER.

Output a valid JSON object strictly matching:
{{
  "total_score": float,
  "sub_scores": {{"content": float, "language": float, "structure": float}},
  "reasoning_trace": "Link the score to components in Z. Specifically justify the assigned tier and the nearest upper/lower boundary.",
  "feedback": "Detailed justification in Chinese focusing on communicative effectiveness."
}}
"""

EXTRACTION_SYSTEM_PROMPT = """You are the Scoring Rubric Component Extraction Agent. 
The student is a Chinese High School Senior (Grade 12). 
Your objective is to extract rubric-relevant evidence from the student response (R). 
As a neutral observer, identify specific components (Boolean flags, counts, and text spans) while understanding common language patterns of Chinese EFL (English as a Foreign Language) learners. 
Do NOT assign scores. Focus exclusively on identifying the presence of rubric-defined elements.
Be concise. Do not provide lengthy reasoning. Output the JSON directly.
"""

SCORING_SYSTEM_PROMPT = """You are the Scoring Agent, acting as a Senior English Examiner for the Chinese National College Entrance Examination (Gaokao).
The students are Chinese Grade 12 learners. You must evaluate the response based on the "Communicative Effectiveness" standard appropriate for this educational stage. You should strictly follow the assigned score to grade.
Be concise. Do not provide lengthy reasoning. Output the JSON directly.
"""


# ---------------------------------------------- #
PURE_TEMPLATE = """
### Task Context (X)
{task_rubric}

### Adaptation Rules (Gar)
{extra_rules}

### Digitization Context (OCR Awareness)
[SYSTEM NOTE]: The following student response has been processed via an OCR (Optical Character Recognition) system. Be advised that the text may contain digitization noise, character misidentifications, or formatting artifacts inherent to the scanning and extraction process. You MUST prioritize the student's communicative intent and overall linguistic proficiency. Do not penalize the score for localized character-level inconsistencies that could reasonably be attributed to digitization noise rather than the student's actual writing.

### Original Response (R)
{essay_text}

### Current TIER
{tier}

Output a valid JSON object strictly matching:
{{
  "total_score": float,
  "sub_scores": {{"content": float, "language": float, "structure": float}},
  "reasoning_trace": "Link the score to rubric evidence. Specifically justify the assigned tier and the nearest upper/lower boundary.",
  "feedback": "Detailed justification in Chinese focusing on communicative effectiveness."
}}

"""
TIER_SYSTEM_PROMPT = """
# ROLE: Senior English Examination Expert (Gaokao Specialist)
# TASK: Assign a TIER (1-{tier_count}) to the student's essay based on HOLISTIC EVALUATION.

【要点评分规则】
1. 逐项核对：先列出题目要求的全部要点清单，逐项检查作文是否覆盖
2. 区分主次：标注哪些是“主要内容”（缺一不可），哪些是“次重点”（可漏1-2个）.学生提供的要点在语法上是相关即可,无需探究现实逻辑.
3. 展开标准：“覆盖”意味着该要点被具体描述（有细节/原因/方式/结果），而非仅出现关键词.
4. 无关内容过滤：统计字数时，先剔除与要点无关的内容
5. 定档依据：
   - 缺任何“主要内容” → 上限为第{tier_count_2}档
   - 要点全齐但大部分仅“提及”未展开 → 上限为第{tier_count_3}档
   - 要点全齐且全部展开 → 才有资格进入第{tier_count_4}/{tier_count}档

Let's think step by step!
"""

TIER_CLASSIFIER_PROMPT = """
### Task
Classify the provided student English response into a writing tier.

### Score Scale
Total Score: {max_score}
Total Tiers: {tier_count}

### Official Rubric
{rubric}

### Student Response
Document ID: {file_name}
{text}

### Evaluation Rules

Classify the response into a coarse tier only. Do not assign a final numeric score.

Accept valid alternative ideas, examples, and expressions that satisfy the task requirements.
Important boundary rules:
- If cause, impact, and relief method are all covered and the message is clear, do not assign Tier 3 merely because of minor grammar, spelling, punctuation, or collocation errors.
- Minor language errors should lower the tier only when they affect communication or show clear inability
 to complete the task.
- Your tier_reason must refer only to observable evidence in the student response: content coverage, clarity, coherence, and language quality.
{{
  "tier": int,
  "confidence": float,
  "content_coverage": {{
  }},
  "reason": str,
  "borderline_with": int or null
}}
"""


BASELINE_PROMPT = """
You are an expert English teacher. Please grade the following student essay based on the provided rubric.

[SCORING RUBRIC]
{rubric}

[STUDENT ESSAY]
{essay_text}

[INSTRUCTION]
1. Read the rubric carefully.
2. Evaluate the essay objectively.
3. Output your response in the following JSON format:
{{
  "reasoning_trace": "Your brief explanation for the score",
  "total_score": X.X
}}
"""
