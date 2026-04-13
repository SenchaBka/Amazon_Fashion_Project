"""Phase 2 Step 7 utilities – Parts B & C: local LLM tasks.

Part B – Summarization
    Selects 10 reviews longer than 100 words from the Phase 2 subset and
    summarises each to approximately 50 words using a local Hugging Face
    summarization pipeline.

Part C – Customer-service response generation
    Finds one question-like review (contains "?" or common question phrases)
    and generates a polite, professional customer-service reply using a local
    Hugging Face text-to-text pipeline.

Both model names are exposed as module-level constants and can be overridden
via CLI arguments in ``run_phase2_step7.py``.
"""

import os

import pandas as pd

# ── Configurable model names ───────────────────────────────────────────────────
# Part B: compact distilled BART fine-tuned on CNN/DailyMail — runs well on CPU.
SUMMARIZATION_MODEL = "sshleifer/distilbart-cnn-12-6"

# Part C: FLAN-T5 base is instruction-following and works via text2text-generation.
#         It is small (~250 MB) and runs on CPU without issue.
GENERATION_MODEL = "google/flan-t5-base"

# Fallback for environments where only 'text-generation' pipelines are available.
# This keeps fallback lightweight and avoids re-downloading large summarization
# checkpoints for an incompatible task.
TEXTGEN_FALLBACK_MODEL = "distilgpt2"

# ── Summarization settings ────────────────────────────────────────────────────
N_LONG_REVIEWS = 10
MIN_WORD_COUNT = 100
TARGET_SUMMARY_WORDS = 50   # ~50-word output target
SUMMARY_MAX_LEN = 70        # token ceiling fed to the model
SUMMARY_MIN_LEN = 30        # token floor

# ── Question-detection heuristics ─────────────────────────────────────────────
QUESTION_PATTERNS = [
    r"\?",
    r"\bdoes this\b",
    r"\bis this\b",
    r"\bcan i\b",
    r"\bwill it\b",
    r"\bhow do i\b",
    r"\bdo these\b",
    r"\bwould this\b",
]


def _load_pipeline(task: str, model: str):
    """Return a Hugging Face ``pipeline`` object, raising a clear error if
    ``transformers`` is not installed.
    """
    try:
        from transformers import pipeline  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "The 'transformers' package is required for Step 7 LLM tasks.\n"
            "Install it with:  pip install transformers sentencepiece torch"
        ) from exc

    print(f"[llm] Loading model '{model}' for task '{task}' "
          "(first run may download model weights) ...")
    return pipeline(task, model=model)


def _try_load_pipeline(task: str, model: str):
    """Try loading a pipeline; return None if task is unsupported."""
    try:
        return _load_pipeline(task, model)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "Unknown task" in msg:
            print(f"[llm] Task '{task}' is unavailable in this transformers build.")
            return None
        raise


def _load_textgen_with_fallback(preferred_model: str):
    """Load a text-generation pipeline from preferred model, else fallback model."""
    # Always use the dedicated lightweight causal-LM fallback model for this task.
    # This avoids expensive/incorrect cross-task loads (e.g. T5/BART checkpoints).
    if preferred_model != TEXTGEN_FALLBACK_MODEL:
        print(
            f"[llm] Using lightweight text-generation model '{TEXTGEN_FALLBACK_MODEL}' "
            f"instead of '{preferred_model}'."
        )
    return _load_pipeline("text-generation", TEXTGEN_FALLBACK_MODEL), TEXTGEN_FALLBACK_MODEL


def _extract_generated_suffix(full_text: str, prompt: str) -> str:
    """Return model completion text when text-generation echoes the prompt."""
    out = str(full_text).strip()
    prefix = str(prompt).strip()
    if out.startswith(prefix):
        return out[len(prefix):].strip()
    return out


def _clean_generated_text(generated_text: str) -> str:
    """Remove prompt-like artifacts and keep concise assistant-style output."""
    text = str(generated_text).strip()

    # Trim common prompt scaffolding that may be echoed by causal models.
    markers = [
        "Summarize the following customer review in about 50 words:",
        "Review:",
        "Summary:",
        "You are a professional customer service representative",
        "Write a polite, concise, and helpful response to this customer.",
    ]
    for marker in markers:
        if marker in text and text.startswith(marker):
            # keep only trailing content after the last marker occurrence
            parts = text.split(marker)
            text = parts[-1].strip()

    # If model leaves empty/near-empty output after cleaning, keep original.
    if len(text) < 8:
        return str(generated_text).strip()
    return text


def _generate_with_fallback(prompt: str, model_name: str, max_new_tokens: int = 150) -> tuple[str, str]:
    """Generate text using text2text when available, else fallback to text-generation.

    Returns
    -------
    tuple[str, str]
        (generated_text, pipeline_task_used)
    """
    generator = _try_load_pipeline("text2text-generation", model_name)
    if generator is not None:
        result = generator(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        return result[0]["generated_text"].strip(), "text2text-generation"

    print("[llm] Falling back to 'text-generation' for response generation.")
    generator, used_model = _load_textgen_with_fallback(model_name)
    result = generator(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    generated = _extract_generated_suffix(result[0]["generated_text"], prompt)
    generated = _clean_generated_text(generated)
    return generated, f"text-generation ({used_model})"


# ── Part B ─────────────────────────────────────────────────────────────────────

def run_step7_summarization(
    subset_csv_path: str,
    out_dir: str,
    model_name: str = SUMMARIZATION_MODEL,
    n_reviews: int = N_LONG_REVIEWS,
    min_words: int = MIN_WORD_COUNT,
) -> dict:
    """Select long reviews and summarise each to ~50 words.

    Parameters
    ----------
    subset_csv_path:
        Path to the Step 1 labeled subset CSV.
    out_dir:
        Directory where output artifacts are written.
    model_name:
        Hugging Face model identifier for the summarization pipeline.
    n_reviews:
        How many long reviews to select (default 10).
    min_words:
        Minimum word count threshold for "long" review (default 100).

    Returns
    -------
    dict
        Summary dict included in the Step 7 overall summary.
    """
    os.makedirs(out_dir, exist_ok=True)

    print(f"[llm] Loading subset from '{subset_csv_path}' ...")
    df = pd.read_csv(subset_csv_path)

    if "text" not in df.columns:
        raise KeyError("Column 'text' not found in subset CSV.")

    # Use pre-computed word_len if available, otherwise compute it
    if "word_len" in df.columns:
        df["_word_count"] = pd.to_numeric(df["word_len"], errors="coerce").fillna(0).astype(int)
    else:
        df["_word_count"] = df["text"].str.split().str.len()

    long_df = df[df["_word_count"] > min_words].reset_index().rename(
        columns={"index": "review_id"}
    )

    available = len(long_df)
    if available == 0:
        raise ValueError(
            f"No reviews with more than {min_words} words found in '{subset_csv_path}'."
        )
    if available < n_reviews:
        print(
            f"[llm] WARNING: Only {available} reviews exceed {min_words} words; "
            f"selecting all {available} instead of {n_reviews}."
        )
        n_reviews = available

    selected = long_df.head(n_reviews).copy()
    print(f"[llm] Selected {len(selected)} reviews with >{min_words} words.")

    # ── Save step7_long_reviews.csv ────────────────────────────────────────────
    long_reviews_out = selected[["review_id", "text", "_word_count"]].copy()
    long_reviews_out.columns = ["review_id", "original_text", "word_count"]
    long_path = os.path.join(out_dir, "step7_long_reviews.csv")
    long_reviews_out.to_csv(long_path, index=False)
    print(f"[llm] Saved: {long_path}")

    # ── Run summarisation ──────────────────────────────────────────────────────
    summarizer = _try_load_pipeline("summarization", model_name)
    summarization_task_used = "summarization" if summarizer is not None else "text-generation"
    fallback_generator = None
    fallback_model_used = None
    if summarizer is None:
        print("[llm] Falling back to 'text-generation' for summarization.")
        # Use lightweight fallback model directly to avoid downloading large
        # summarization checkpoints for incompatible text-generation usage.
        fallback_generator = _load_pipeline("text-generation", TEXTGEN_FALLBACK_MODEL)
        fallback_model_used = TEXTGEN_FALLBACK_MODEL

    generated_summaries: list[str] = []
    for idx, (_, row) in enumerate(selected.iterrows()):
        text = str(row["text"])
        print(
            f"[llm] Summarising review {idx + 1}/{len(selected)} "
            f"(review_id={row['review_id']}, {row['_word_count']} words) ..."
        )
        try:
            if summarizer is not None:
                result = summarizer(
                    text,
                    max_length=SUMMARY_MAX_LEN,
                    min_length=SUMMARY_MIN_LEN,
                    do_sample=False,
                    truncation=True,
                )
                summary_text = result[0]["summary_text"].strip()
            else:
                prompt = (
                    "Summarize the following customer review in about 50 words.\n\n"
                    f"Review:\n{text}\n\nSummary:"
                )
                result = fallback_generator(
                    prompt,
                    max_new_tokens=120,
                    do_sample=False,
                )
                summary_text = _extract_generated_suffix(result[0]["generated_text"], prompt)
                summary_text = _clean_generated_text(summary_text)
        except Exception as exc:  # noqa: BLE001
            print(f"[llm] WARNING: Summarisation failed for review_id={row['review_id']}: {exc}")
            summary_text = ""
        generated_summaries.append(summary_text)

    # ── Save step7_summaries.csv ───────────────────────────────────────────────
    summaries_df = pd.DataFrame(
        {
            "review_id": selected["review_id"].values,
            "original_text": selected["text"].values,
            "word_count": selected["_word_count"].values,
            "generated_summary": generated_summaries,
        }
    )
    summaries_path = os.path.join(out_dir, "step7_summaries.csv")
    summaries_df.to_csv(summaries_path, index=False)
    print(f"[llm] Saved: {summaries_path}")

    return {
        "long_reviews_available": available,
        "reviews_selected": int(len(selected)),
        "min_word_threshold": min_words,
        "model": model_name,
        "pipeline_task_used": summarization_task_used,
        "fallback_model_used": fallback_model_used,
        "target_summary_words": TARGET_SUMMARY_WORDS,
        "artifacts": {
            "long_reviews": long_path,
            "summaries": summaries_path,
        },
    }


# ── Part C ─────────────────────────────────────────────────────────────────────

def run_step7_response_generation(
    subset_csv_path: str,
    out_dir: str,
    model_name: str = GENERATION_MODEL,
) -> dict:
    """Find one question-like review and generate a customer-service response.

    Parameters
    ----------
    subset_csv_path:
        Path to the Step 1 labeled subset CSV.
    out_dir:
        Directory where output artifacts are written.
    model_name:
        Hugging Face model identifier for the text-generation pipeline.

    Returns
    -------
    dict
        Summary dict included in the Step 7 overall summary.
    """
    os.makedirs(out_dir, exist_ok=True)

    print(f"[llm] Loading subset from '{subset_csv_path}' ...")
    df = pd.read_csv(subset_csv_path)

    if "text" not in df.columns:
        raise KeyError("Column 'text' not found in subset CSV.")

    # ── Detect question-like reviews ───────────────────────────────────────────
    pattern = "|".join(QUESTION_PATTERNS)
    question_mask = df["text"].str.lower().str.contains(pattern, regex=True, na=False)
    question_df = df[question_mask].copy()

    if question_df.empty:
        raise ValueError(
            "No question-like review found with required markers ('?' or allowed phrases)."
        )

    # Keep only clean, non-empty candidates and pick the first deterministic match.
    question_df = question_df[question_df["text"].astype(str).str.strip().str.len() > 5]
    if question_df.empty:
        raise ValueError("Question-like candidates were found, but all were empty/invalid.")

    selected_row = question_df.iloc[0]
    review_id = int(selected_row.name)
    review_text = str(selected_row["text"])

    print(f"[llm] Selected question review (review_id={review_id}, "
          f"{len(review_text.split())} words).")

    # ── Save step7_question_review.csv ─────────────────────────────────────────
    question_out = pd.DataFrame(
        [
            {
                "review_id": review_id,
                "original_text": review_text,
                "word_count": len(review_text.split()),
            }
        ]
    )
    question_path = os.path.join(out_dir, "step7_question_review.csv")
    question_out.to_csv(question_path, index=False)
    print(f"[llm] Saved: {question_path}")

    # ── Build prompt ───────────────────────────────────────────────────────────
    prompt = (
        "You are a professional customer service representative for an online "
        "fashion retailer. A customer has left the following review or question:\n\n"
        f'"{review_text}"\n\n'
        "Write a polite, concise, and helpful response to this customer."
    )

    # ── Generate response ──────────────────────────────────────────────────────
    print(f"[llm] Generating response with '{model_name}' ...")
    try:
        response_text, generation_task_used = _generate_with_fallback(
            prompt=prompt,
            model_name=model_name,
            max_new_tokens=150,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[llm] ERROR during response generation: {exc}")
        raise

    # ── Save step7_generated_response.txt ──────────────────────────────────────
    response_path = os.path.join(out_dir, "step7_generated_response.txt")
    with open(response_path, "w", encoding="utf-8") as fh:
        fh.write("=== Selected Customer Review ===\n\n")
        fh.write(review_text + "\n\n")
        fh.write("=== Generated Customer Service Response ===\n\n")
        fh.write(response_text + "\n")
    print(f"[llm] Saved: {response_path}")

    return {
        "review_id": review_id,
        "review_preview": review_text[:120] + ("..." if len(review_text) > 120 else ""),
        "response_preview": response_text[:120] + ("..." if len(response_text) > 120 else ""),
        "model": model_name,
        "pipeline_task_used": generation_task_used,
        "artifacts": {
            "question_review": question_path,
            "generated_response": response_path,
        },
    }
