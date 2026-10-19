"""
question_generator.py — T5-based MCQ generator from PDF text.

Pipeline:
  1. Extract text from PDF with pdfplumber
  2. Split into sentences, extract answer candidates (noun phrases, numbers)
  3. Feed each (answer, context) pair to valhalla/t5-small-qg-prepend
  4. Build 4-option MCQ using document-extracted distractors
  5. Assign difficulty level 1/2/3 by sentence word-count
"""

import logging
import re
import time
import uuid
import random
import torch
import pdfplumber
from huggingface_hub import snapshot_download
from transformers import T5ForConditionalGeneration, T5Tokenizer

log = logging.getLogger(__name__)

# Roman numeral detector — rejects I, II, III, IV, V, X, XI … as answer candidates
_ROMAN_RE = re.compile(
    r'^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$',
    re.IGNORECASE,
)

def _is_roman(text: str) -> bool:
    t = text.strip()
    return len(t) >= 1 and bool(_ROMAN_RE.fullmatch(t))

MODEL_NAME = "valhalla/t5-small-qg-prepend"

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_t5_from_hub(model_id: str) -> tuple[T5Tokenizer, T5ForConditionalGeneration]:
    """
    Download with a single worker (avoids flaky parallel httpx on some Windows setups),
    then load strictly from disk. Retries the known “client has been closed” race.
    """
    last: BaseException | None = None
    for attempt in range(1, 4):
        try:
            repo_dir = snapshot_download(
                repo_id=model_id,
                max_workers=1,
            )
            tok = T5Tokenizer.from_pretrained(repo_dir, local_files_only=True)
            mdl = T5ForConditionalGeneration.from_pretrained(repo_dir, local_files_only=True)
            return tok, mdl
        except RuntimeError as e:
            last = e
            low = str(e).lower()
            if "closed" in low or "client" in low:
                wait = 2.0 * attempt
                log.warning("HF load attempt %s/3 hit httpx client issue: %s — sleeping %ss", attempt, e, wait)
                time.sleep(wait)
                continue
            raise
        except OSError as e:
            last = e
            if attempt < 3:
                wait = 2.0 * attempt
                log.warning("HF snapshot/load OSError (%s); retry in %ss", e, wait)
                time.sleep(wait)
                continue
            raise
    assert last is not None
    raise RuntimeError(f"Could not load T5 after 3 attempts: {last}") from last


class QuestionGenerator:
    def __init__(self):
        self.tokenizer, model = _load_t5_from_hub(MODEL_NAME)
        # int8 dynamic quantization — ~2x faster on CPU, negligible quality loss
        # Apple Silicon/ARM ships qnnpack but leaves the engine unset ("NoQEngine").
        engines = torch.backends.quantized.supported_engines
        if torch.backends.quantized.engine == "none" and "qnnpack" in engines:
            torch.backends.quantized.engine = "qnnpack"
        if _DEVICE.type == "cpu" and torch.backends.quantized.engine != "none":
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        self.model = model.to(_DEVICE)
        self.model.eval()

    # ── PDF ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _is_roman_numbered_page(page_text: str) -> bool:
        """
        Return True if this page uses Roman numeral page numbering (i, ii, iii, iv …).
        Detected by finding a line that contains ONLY a Roman numeral — typical of
        front-matter pages (cover, preface, TOC, acknowledgements) in academic PDFs.
        Both lowercase (iv) and uppercase (IV) are detected.
        """
        for line in page_text.split("\n"):
            stripped = line.strip()
            if stripped and _is_roman(stripped):
                return True
        return False

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if not page_text:
                    continue
                # Skip front-matter pages — those numbered with Roman numerals
                if self._is_roman_numbered_page(page_text):
                    continue
                text += page_text + " "
        return re.sub(r"\s+", " ", text).strip()

    # ── Text helpers ──────────────────────────────────────────────────────────

    def split_sentences(self, text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in parts if len(s.split()) >= 6]

    def extract_answer_candidates(self, sentences: list[str]) -> list[tuple]:
        """Return list of (answer_str, source_sentence) pairs."""
        STOPWORDS = {
            "the", "a", "an", "in", "on", "at", "of", "to", "and", "or", "is", "are",
            "per", "for", "with", "by", "from", "as",
        }
        candidates = []
        seen: set[str] = set()
        for sentence in sentences:
            for phrase in re.findall(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b", sentence):
                if (phrase.lower() not in STOPWORDS
                        and phrase not in seen
                        and not _is_roman(phrase)):       # skip Roman numerals
                    seen.add(phrase)
                    candidates.append((phrase, sentence))
            for phrase in re.findall(r'"([^"]{3,40})"', sentence):
                if phrase not in seen and not _is_roman(phrase):
                    seen.add(phrase)
                    candidates.append((phrase, sentence))
            for phrase in re.findall(r"\b\d+(?:[.,]\d+)?(?:\s+\w+){0,2}\b", sentence):
                stripped = phrase.strip()
                # The {0,2} trailing-word match can grab a stopword right before the
                # sentence's next clause (e.g. "206 bones in", "300000 kilometers per") —
                # trim those off so the extracted answer ends on a real word.
                words = stripped.split()
                while len(words) > 1 and words[-1].lower() in STOPWORDS:
                    words.pop()
                stripped = " ".join(words)
                if stripped not in seen and len(stripped) > 1 and not _is_roman(stripped):
                    seen.add(stripped)
                    candidates.append((stripped, sentence))
        return candidates

    # ── Question generation ───────────────────────────────────────────────────

    MINI_BATCH = 32

    def generate_questions_batch(self, pairs: list[tuple]) -> list[str]:
        if not pairs:
            return []
        all_results: list[str] = []
        for i in range(0, len(pairs), self.MINI_BATCH):
            chunk = pairs[i : i + self.MINI_BATCH]
            # Truncate context to 250 chars — enough for T5 to form a question,
            # shorter sequences mean faster encoder forward pass
            input_texts = [
                f"answer: {ans} context: {ctx[:250]}"
                for ans, ctx in chunk
            ]
            encoded = self.tokenizer(
                input_texts,
                return_tensors="pt",
                max_length=256,
                truncation=True,
                padding=True,
            ).to(_DEVICE)
            with torch.inference_mode():
                outputs = self.model.generate(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    max_new_tokens=32,
                    num_beams=1,
                    do_sample=False,
                )
            all_results.extend(
                self.tokenizer.decode(out, skip_special_tokens=True).strip()
                for out in outputs
            )
        return all_results

    def generate_question(self, answer: str, context: str) -> str:
        """Single question generation (kept for compatibility)."""
        results = self.generate_questions_batch([(answer, context)])
        return results[0] if results else ""

    # ── Distractor generation ─────────────────────────────────────────────────

    def build_distractor_pool(self, sentences: list[str]) -> list[str]:
        pool: list[str] = []
        seen: set[str] = set()
        for sentence in sentences:
            for phrase in re.findall(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b", sentence):
                if phrase not in seen and not _is_roman(phrase):
                    seen.add(phrase)
                    pool.append(phrase)
        return pool

    def pick_distractors(self, correct: str, pool: list[str], n: int = 3) -> list[str]:
        correct_lower = correct.lower()
        correct_words = correct_lower.split()
        correct_first = correct_words[0] if correct_words else correct_lower
        correct_len   = len(correct_words)

        def is_valid(phrase: str) -> bool:
            pl = phrase.lower()
            pw = pl.split()
            if pl == correct_lower:
                return False
            # Reject if first word matches — student can eliminate immediately
            if pw and pw[0] == correct_first:
                return False
            # Reject substring/superstring pairs — too visually similar
            if correct_lower in pl or pl in correct_lower:
                return False
            return True

        valid = [p for p in pool if is_valid(p)]

        # Prefer distractors with similar word count (±2) — more plausible MCQ options
        similar_len = [p for p in valid if abs(len(p.split()) - correct_len) <= 2]
        candidates  = similar_len if len(similar_len) >= n else valid

        random.shuffle(candidates)
        distractors = candidates[:n]
        fallbacks = ["None of the above", "All of the above", "Cannot be determined"]
        i = 0
        while len(distractors) < n:
            distractors.append(fallbacks[i % len(fallbacks)])
            i += 1
        return distractors

    # ── Main entry point ──────────────────────────────────────────────────────

    def generate_from_pdf(
        self, pdf_path: str, num_questions: int = 9, concept_tag: str = "General"
    ) -> dict:
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            raise ValueError("Could not extract text from PDF.")

        sentences      = self.split_sentences(text)
        candidates     = self.extract_answer_candidates(sentences)
        distractor_pool = self.build_distractor_pool(sentences)

        per_level = max(1, num_questions // 3)   # 3 for a 9-question assignment
        needed    = per_level * 3                 # always a clean multiple of 3

        # Gather 3× as many candidates as needed so we have enough after filtering
        unique_candidates: list[tuple] = []
        seen_ctx: set[str] = set()
        for answer, context in candidates:
            if len(unique_candidates) >= needed * 3:
                break
            if context not in seen_ctx:
                unique_candidates.append((answer, context))
                seen_ctx.add(context)

        if not unique_candidates:
            raise ValueError("No answer candidates found in PDF.")

        # T5 batch inference
        q_texts = self.generate_questions_batch(unique_candidates)

        # Build question objects (no level assigned yet)
        questions: list[dict] = []
        for (answer, context), q_text in zip(unique_candidates, q_texts):
            if not q_text or len(q_text) < 8:
                continue
            distractors = self.pick_distractors(answer, distractor_pool)
            options = [answer] + distractors
            random.shuffle(options)
            questions.append({
                "questionId":             str(uuid.uuid4())[:8],
                "questionText":           q_text,
                "options":                options[:4],
                "correctAnswer":          answer,
                "hint":                   context[:160].strip(),
                "shortTheoryExplanation": context.strip(),
                "conceptTag":             concept_tag,
                "_ctx_len":               len(context.split()),  # temp field for sorting
            })

        if not questions:
            raise ValueError("T5 produced no usable questions from this PDF.")

        # Sort by context length (shortest = simplest = Level 1, longest = hardest = Level 3)
        # then take up to `needed` questions and split into 3 roughly-even levels by actual
        # count — if fewer than `needed` survived filtering, levels stay balanced (e.g. 3/2/2)
        # instead of front-loading Level 1 with a fixed per-level size.
        questions.sort(key=lambda q: q["_ctx_len"])
        questions = questions[:needed]
        actual = len(questions)

        for idx, q in enumerate(questions):
            q["quizLevel"] = min(3, idx * 3 // actual + 1)
            del q["_ctx_len"]

        bucket: dict[int, list] = {1: [], 2: [], 3: []}
        for q in questions:
            bucket[q["quizLevel"]].append(q)

        return {"level1": bucket[1], "level2": bucket[2], "level3": bucket[3]}
