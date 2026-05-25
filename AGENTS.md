# Project Instructions for Codex

## Very important dataset rule

Do NOT open, cat, read, parse, grep, or load `dataset.json` directly unless the user explicitly says so.

`dataset.json` is large. Treat this file as a schema description only. Use the schema below as the source of truth for how to write code against the dataset.

If you need to inspect example data, read `dataset_sample_schema.json`, not `dataset.json`.

## Dataset structure

The dataset is a normal JSON object.

Top-level keys are paper IDs, for example:

    English_Grade12_001
    English_Grade12_002
    English_Grade12_003
    English_Grade12_004
    English_Grade12_005
    English_Grade12_006

Each top-level value is one exam paper package.

A paper object has this structure:

    paper = {
        "subject": "English",

        "blank_sheet": {
            "66": "English_Grade12_001/Blank/66.tif",
            "67": "English_Grade12_001/Blank/67.tif"
        },

        "subjective_question": {
            "66": {
                "question": "...",
                "rubric": "...",
                "reference_answer": "...",
                "reference_analysis": "..."
            },
            "67": {
                "question": "...",
                "rubric": "...",
                "reference_answer": "...",
                "reference_analysis": "..."
            }
        },

        "student_answer": [
            {
                "id": "307011440",
                "answer": {
                    "66": "English_Grade12_001/Scan/307011440_66.tif",
                    "67": "English_Grade12_001/Scan/307011440_67.tif"
                },
                "grading": {
                    "66": 10.0,
                    "67": 16.5
                }
            }
        ]
    }

## Important field notes

- Student answers are image paths, not OCR text.
- Question `66` is usually a 15-point writing task.
- Question `67` is usually a 25-point continuation-writing task.
- `student_answer[*].grading[qid]` is the human score.
- `student_answer[*].answer[qid]` is the scanned answer image path.
- `blank_sheet[qid]` is the blank template image path.
- Some question objects may have `reference_analysis`; code should use `.get("reference_analysis")` instead of assuming it always exists.

## Recommended flattened sample format

When writing processing code, flatten each student-question pair into this logical format:

    {
        "paper_id": "...",
        "subject": "...",
        "student_id": "...",
        "question_id": "66",
        "question": "...",
        "rubric": qinfo["rubric"],
        "reference_answer": "...",
        "reference_analysis": "...",
        "blank_sheet_path": "...",
        "student_answer_path": "...",
        "score": 10.0
    }

## Safe access pattern

Prefer writing code that accepts a dataset path but does not inspect the full file during reasoning.

Correct general iteration pattern:

    for paper_id, paper in data.items():
        subject = paper.get("subject")
        blank_sheet = paper.get("blank_sheet", {})
        questions = paper.get("subjective_question", {})
        students = paper.get("student_answer", [])

        for student in students:
            student_id = student.get("id")
            answers = student.get("answer", {})
            grading = student.get("grading", {})

            for qid, image_path in answers.items():
                qinfo = questions.get(str(qid), {})
                score = grading.get(str(qid))

                sample = {
                    "paper_id": paper_id,
                    "subject": subject,
                    "student_id": student_id,
                    "question_id": str(qid),
                    "question": qinfo.get("question"),
                    "rubric": qinfo.get("rubric"),
                    "reference_answer": qinfo.get("reference_answer"),
                    "reference_analysis": qinfo.get("reference_analysis"),
                    "blank_sheet_path": blank_sheet.get(str(qid)),
                    "student_answer_path": image_path,
                    "score": score,
                }

## Do not do this

Do not write commands or code that inspect the real dataset during development, such as:

    cat dataset.json
    head dataset.json
    tail dataset.json
    grep something dataset.json
    python -c "import json; json.load(open('dataset.json'))"

Only load `dataset.json` inside final runtime scripts when the user explicitly runs them.