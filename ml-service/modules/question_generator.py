"""
question_generator.py — T5-based MCQ generator from PDF text.

Pipeline:
  1. Extract text from PDF with pdfplumber, dropping page furniture (repeated
     headers/footers, page numbers, figure labels, garbled legacy-font lines)
     and collecting short heading lines as the document's key terms
  2. Split into sentences and keep factual ones, spread across the whole document
  3. Pick a meaningful answer per sentence: a defined term ("... is called X"),
     a key term, or a number — never a sentence-initial word like "This"/"Use"
  4. Feed each (answer, context) pair to valhalla/t5-small-qg-prepend
  5. Reject unusable questions (no "?", answer leaked into the question, ...)
  6. Build 4-option MCQ with same-type distractors (term → nearby terms,
     number → nearby numbers)
  7. Assign difficulty level 1/2/3 by sentence word-count
"""

import bisect
import difflib
import logging
import math
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


# ── Text-cleaning / answer-selection patterns ─────────────────────────────────

_BULLET_RE      = re.compile(r"^[^\w\s(\"'“]+\s*")          # leading •, ², � …
_SECTION_NUM_RE = re.compile(r"^\d+(?:\.\d+)*\.?\s+")        # "1.1.1 ", "3. "
_SECTION_HEADING_RE = re.compile(r"^\d{1,2}\.\d{1,2}\s+[A-Za-z]")  # "1.1 Plant tissues" (one level)
_SENT_SPLIT_RE  = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])")
_TERMINAL       = (".", "?", "!", ":", ";", ",", "-")

# "... is called X." / "... are known as X," → X is the defined term
_DEF_RE = re.compile(
    r"\b(?:is|are|was|were)\s+(?:also\s+)?(?:called|known\s+as|referred\s+to\s+as|termed|named)\s+"
    r"(?:an?\s+|the\s+)?([A-Za-z][A-Za-z\- ]{1,50}?)\s*"
    r"(?:[.,;:(]|\band\b|\bor\b|\bwhile\b|\bwhereas\b|\bbut\b|\bwhich\b|\bthat\b|$)",
    re.IGNORECASE,
)
# Heading prefixes that wrap the actual term: "Features of meristematic tissues"
_HEADING_PREFIX_RE = re.compile(
    r"^(?:features|functions|locations|types|uses|properties|process|classification|importance|"
    r"applications|structure|parts|characteristics|composition|separation|production|regulation|"
    r"examples|diseases associated with)\s+of\s+(?:the\s+|an?\s+)?"
    r"|^factors\s+(?:that\s+)?affect(?:ing|s)?\s+",
    re.IGNORECASE,
)
_NUM_RE = re.compile(
    r"(?<![\w.×-])(\d+(?:\.\d+)?)"
    r"(\s?(?:%|°C|0C|mmHg|mm|cm|km|kg|mg|g|Hz|days|hours|minutes|seconds|years|million|billion|m|s)\b)?"
    r"(?![\w.×^-])"
)
_NUM_SKIP_BEFORE_RE = re.compile(
    r"(figure|fig\.|activity|assignment|table|grade|page|chapter|unit|step|exercise|part)\s*$", re.I
)
# Sentences that only make sense next to a figure/activity, or are instructions
_NOISE_RE = re.compile(
    r"\b(figure|fig\.|activity|assignment|diagram|exercise|table|following|given below|above|shown|"
    r"let us|let's|below|illustrated|this (?:way|method|process|case|reason))\b",
    re.IGNORECASE,
)
_SKIP_HEADING_WORDS = {
    "activity", "assignment", "figure", "table", "exercise", "exercises", "summary", "glossary",
    "technical", "method", "materials", "extra", "knowledge", "note", "contents", "page",
    "example", "examples", "worked", "introduction", "foreword", "answer", "for", "free", "distribution",
    "biology", "chemistry", "physics", "science",
}
_PRONOUN_START = {
    "this", "these", "that", "those", "it", "its", "they", "their", "he", "she", "we", "you",
    "such", "there", "here", "then", "also", "so", "but", "and", "or", "hence", "therefore",
    "thus", "however", "accordingly", "next", "now",
}
_QUESTION_STARTS = {
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "in", "on", "at",
    "for", "by", "from", "to", "during", "after", "before", "is", "are", "does", "do", "did",
    "can", "name", "into", "through", "with", "about", "under", "between", "the", "a", "an",
}
_STOPWORDS = {
    "the", "a", "an", "in", "on", "at", "of", "to", "and", "or", "is", "are", "was", "were",
    "per", "for", "with", "by", "from", "as", "be", "been", "this", "that", "these", "those",
    "it", "its", "their", "which", "who", "what", "has", "have", "had", "not", "no", "can",
    "into", "than", "then", "also", "such", "some", "all", "any", "each", "other", "more", "most",
}
_FALLBACK_DISTRACTORS = ["None of the above", "All of the above", "Cannot be determined"]


def _stem(word: str) -> str:
    """Crude singular form so "meristems"/"meristem" and "tissues"/"tissue" match."""
    w = word.lower()
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _term_pattern(term: str) -> re.Pattern:
    """Word-boundary pattern for a term that tolerates a plural/singular last word."""
    words = term.lower().split()
    last = re.escape(_stem(words[-1]))
    head = r"\s+".join(re.escape(w) for w in words[:-1])
    body = (head + r"\s+" if head else "") + last + r"(?:s|es)?"
    return re.compile(r"\b" + body + r"\b", re.IGNORECASE)


def _is_garbled(line: str) -> bool:
    """Legacy Sinhala/Tamil font text extracts as junk like 'úNdcl mgl ¤›°øÇ¯®'."""
    if re.search(r"[a-z];[a-z]", line):
        return True
    ok = sum(1 for c in line if c.isascii() and (c.isalnum() or c in " .,;:()'\"-/%+=?!"))
    return ok / max(1, len(line)) < 0.88


# ── Chapter location (whole textbook uploaded → use only the chosen chapter) ──

# Book chapter titles that differ from the syllabus wording the app uses
_CHAPTER_TITLE_ALIASES = {"organizational levels and interactions of biosphere": "Biosphere"}
_CHAPTER_TOP_LINES = 4      # chapter opener / running header sits in a page's first lines
_CHAPTER_KEY_LEN = 18       # openers wrap long titles over two lines — match the first letters


def _norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _chapter_key(title: str) -> str:
    return _norm_key(_CHAPTER_TITLE_ALIASES.get(title.strip().lower(), title))[:_CHAPTER_KEY_LEN]


def find_chapter_pages(pdf_path: str, title: str, next_title: str = "",
                       prev_title: str = "") -> list[int] | None:
    """0-based page indexes of chapter ``title`` inside a textbook PDF, or None if not found.

    The chapter starts on the first page whose top lines (opener or running header)
    carry its title — searching after ``prev_title``'s opener, so a same-named sub-heading
    in an earlier chapter ("Chemical bonds" inside "Structure of matter") isn't mistaken
    for it — and ends where ``next_title`` starts (or at the end of the PDF).
    Contents pages are skipped. None means "not a textbook with this chapter" — e.g.
    the teacher uploaded just the chapter's own pages — so callers use the whole PDF.
    """
    key, next_key = _chapter_key(title), _chapter_key(next_title) if next_title else ""
    prev_key = _chapter_key(prev_title) if prev_title else ""
    if len(key) < 4:
        return None
    with pdfplumber.open(pdf_path) as pdf:
        tops: list[str | None] = []
        for page in pdf.pages:
            lines = [l.strip() for l in (page.dedupe_chars().extract_text() or "").split("\n") if l.strip()]
            is_contents = sum(bool(re.search(r"\s\d{1,3}\s*$", l)) for l in lines) >= 5
            tops.append(None if is_contents else _norm_key(" ".join(lines[:_CHAPTER_TOP_LINES])))
    after = 0
    if prev_key:
        prev_start = next((i for i, t in enumerate(tops) if t and prev_key in t), None)
        after = prev_start + 1 if prev_start is not None else 0
    start = next((i for i in range(after, len(tops)) if tops[i] and key in tops[i]), None)
    if start is None:
        return None
    end = len(tops)
    if next_key:
        end = next((i for i in range(start + 1, len(tops)) if tops[i] and next_key in tops[i]), end)
    # Found only on the last page, or a 1-page "chapter": treat as not found
    if end - start < 2:
        return None
    return list(range(start, end))


class QuestionGenerator:
    def __init__(self):
        self.tokenizer, model = _load_t5_from_hub(MODEL_NAME)
        # int8 dynamic quantization — ~2x faster on CPU, negligible quality loss
        if _DEVICE.type == "cpu":
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
        Only the first/last lines are checked (where page numbers sit) and a lone
        uppercase letter is ignored, so diagram labels like "C" or "D" on a content
        page don't get the whole page thrown away.
        """
        lines = [l.strip() for l in page_text.split("\n") if l.strip()]
        for stripped in lines[:2] + lines[-2:]:
            if _is_roman(stripped) and (stripped.islower() or len(stripped) >= 2):
                return True
        return False

    def _read_pages(self, pdf_path: str, page_indexes: list[int] | None = None) -> list[list[str]]:
        pages: list[list[str]] = []
        with pdfplumber.open(pdf_path) as pdf:
            selected = pdf.pages if page_indexes is None else [
                pdf.pages[i] for i in page_indexes if 0 <= i < len(pdf.pages)
            ]
            for page in selected:
                # Fake-bold text is drawn twice → "nnuucclleeuuss" without dedupe
                page_text = page.dedupe_chars().extract_text()
                if not page_text or self._is_roman_numbered_page(page_text):
                    continue
                pages.append([l.strip() for l in page_text.split("\n") if l.strip()])
        return pages

    def _extract(self, pdf_path: str, page_indexes: list[int] | None = None
                 ) -> tuple[list[str], list[str], list[str]]:
        """Return (body text chunks, heading lines, section of each chunk).

        Headings/labels are kept out of body text. A chunk's section is the last
        "n.k Title" heading above it (e.g. "Plant tissues"), or "" before the first one.
        """
        pages = self._read_pages(pdf_path, page_indexes)

        # Running headers/footers repeat on many pages ("For free distribution 12")
        counts: dict[str, int] = {}
        for lines in pages:
            for key in {re.sub(r"\d+", "#", l.lower()) for l in lines}:
                counts[key] = counts.get(key, 0) + 1
        furniture = {k for k, c in counts.items() if c >= 3 and c >= len(pages) * 0.05}

        chunks: list[str] = []
        headings: list[str] = []
        sections: list[str] = []                   # parallel to chunks
        current: list[str] = []
        section = ""

        def add(chunk: str):
            chunks.append(chunk)
            sections.append(section)

        def close():
            if current:
                add(" ".join(current))
                current.clear()

        for lines in pages:
            lines = [l for l in lines
                     if re.sub(r"\d+", "#", l.lower()) not in furniture
                     and not re.fullmatch(r"[\d\s.()\-]+", l) and not _is_garbled(l)]
            for i, line in enumerate(lines):
                is_bullet = not line[0].isalnum() and bool(_BULLET_RE.match(line))
                body = _BULLET_RE.sub("", line).strip() if is_bullet else line
                if not body:
                    continue
                title = _SECTION_NUM_RE.sub("", body).strip()
                n_words = len(title.split())
                numbered = title != body
                next_upper = i + 1 >= len(lines) or lines[i + 1][:1].isupper()
                max_words = 12 if numbered else (8 if next_upper else 5)
                if (not is_bullet and 1 <= n_words <= max_words and title[:1].isupper()
                        and not title.endswith(_TERMINAL)):
                    headings.append(title)          # heading or figure label
                    close()
                    # "1.1 Plant tissues" starts a new section (not "1.1.1 …" or "Figure 1.1")
                    if _SECTION_HEADING_RE.match(body) and n_words <= 8:
                        section = re.sub(r"\s+", " ", title).strip(" .:-")
                    continue
                if is_bullet:                       # list item = its own sentence
                    close()
                    add(body if body.endswith((".", "?", "!")) else body + ".")
                    continue
                current.append(line)
        close()
        return chunks, headings, sections

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        chunks, _, _ = self._extract(pdf_path)
        return re.sub(r"\s+", " ", " ".join(chunks)).strip()

    # ── Text helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _sentences(chunks: list[str], sections: list[str]) -> tuple[list[str], list[str]]:
        """Split chunks into sentences; return (sentences, section of each sentence)."""
        out: list[str] = []
        out_sections: list[str] = []
        for chunk, section in zip(chunks, sections):
            for s in _SENT_SPLIT_RE.split(re.sub(r"\s+", " ", chunk).strip()):
                s = re.sub(r"^(?:e\.?g\.?|example)\s*[:;(¡-]*\s*", "", s, flags=re.I).strip()
                if s:
                    out.append(s)
                    out_sections.append(section)
        return out, out_sections

    @staticmethod
    def _usable_context(sentence: str) -> bool:
        words = sentence.split()
        if not (6 <= len(words) <= 40) or not sentence.endswith("."):
            return False
        if words[0].lower().strip(",") in _PRONOUN_START or _NOISE_RE.search(sentence):
            return False
        if len(_DEF_RE.findall(sentence)) >= 2:   # two definitions in one sentence confuse T5
            return False
        if re.search(r"[=�×÷]", sentence):    # broken formula text ("distance � speed = time")
            return False
        alpha = sum(1 for w in words if re.fullmatch(r"[A-Za-z][A-Za-z'\-]*[.,;:]?", w))
        return alpha / len(words) >= 0.7

    def _build_terms(self, headings: list[str], sentences: list[str], body: str) -> dict[str, int]:
        """Key terms of the document → position of first use in the body text."""
        raw: list[str] = []
        for h in headings:
            t = _HEADING_PREFIX_RE.sub("", h).strip(" .:-")
            # A lone word is often a figure label ("West", "Fruit") — trust it less
            raw.append((t, 1.0 if len(t.split()) > 1 else 0.0))
        for s in sentences:
            for m in _DEF_RE.finditer(s):
                raw.append((m.group(1), 2.0))         # defined in the text → a real key term

        body_lower = body.lower()
        terms: dict[str, tuple[int, float]] = {}
        index: dict[str, str] = {}                     # lowercase → stored spelling
        for t, quality in raw:
            t = re.sub(r"\s+", " ", t).strip()
            words = t.split()
            if not (1 <= len(words) <= 4) or len(t) < 3:
                continue
            if not re.fullmatch(r"[A-Za-z][A-Za-z\- ]*", t) or _is_roman(t):
                continue
            if any(len(w) < 2 for w in words):        # "B C", "A a"
                continue
            lw = [w.lower() for w in words]
            if any(w in _SKIP_HEADING_WORDS for w in lw) or all(w in _STOPWORDS for w in lw):
                continue
            if lw[-1] in _STOPWORDS or lw[0] in _STOPWORDS:
                continue
            m = _term_pattern(t).search(body_lower)
            if not m:                               # labels that never appear in prose
                continue
            key = t.lower()
            if key in index:
                old = index[key]
                terms[old] = (terms[old][0], max(terms[old][1], quality))
            else:
                index[key] = t
                terms[t] = (m.start(), quality)
        return terms

    def _pick_answer(self, sentence: str, terms: dict[str, tuple[int, float]]) -> tuple[str, str] | None:
        """Return (answer, kind) where kind is 'term' or 'number', or None."""
        def once(ans: str) -> bool:
            return len(_term_pattern(ans).findall(sentence)) == 1

        # 1) the term this sentence defines
        for m in _DEF_RE.finditer(sentence):
            ans = m.group(1).strip()
            words = ans.split()
            while words and words[-1].lower() in _STOPWORDS:
                words.pop()
            while words and words[0].lower() in _STOPWORDS:   # "… known as as storage tissues"
                words.pop(0)
            ans = " ".join(words)
            if 1 <= len(words) <= 4 and len(ans) >= 3 and once(ans):
                return ans, "term"

        # 2) the longest key term mentioned in the sentence
        n_words = len(sentence.split())
        best = None
        for t in terms:
            m = _term_pattern(t).search(sentence)
            if not m:
                continue
            ans = m.group(0)
            if len(ans.split()) > n_words * 0.5 or not once(ans):
                continue
            if best is None or len(ans) > len(best):
                best = ans
        if best:
            return best, "term"

        # 3) a number (with its unit) that isn't a figure/activity reference
        for m in _NUM_RE.finditer(sentence):
            if _NUM_SKIP_BEFORE_RE.search(sentence[: m.start()]):
                continue
            # A bare number needs a unit or must be a year — "6" alone gives
            # "How many reactions can be obtained?" style questions
            if not m.group(2) and not re.fullmatch(r"1[5-9]\d\d|20\d\d", m.group(1)):
                continue
            ans = m.group(0).strip()
            if sentence.count(m.group(1)) == 1:
                return ans, "number"
        return None

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

    @staticmethod
    def _valid_question(question: str, answer: str) -> bool:
        q = question.strip()
        words = q.split()
        if not q.endswith("?") or len(words) < 5:
            return False
        if words[0].lower().strip(",") not in _QUESTION_STARTS or _NOISE_RE.search(q):
            return False
        long_words = [w.lower().strip("?,") for w in words if len(w.strip("?,")) >= 4]
        if len(long_words) != len(set(long_words)):  # "the composition of the composition"
            return False
        # Figure/apparatus labels only make sense next to the picture: "tube B", "cells 2"
        if re.search(r"\b(?:[A-Z]|\d{1,2})\?*$|\s(?:[A-HJ-Z]|\d{1,2})\s", q) and not re.search(r"\d", answer):
            return False
        # Unfinished questions: "What is the structure of?", "What are of different forms?"
        last = words[-1].lower().rstrip("?")
        prev = words[-2].lower() if len(words) > 1 else ""
        fine_ending = (last == "as" and prev in ("known", "referred", "regarded", "defined", "described",
                                                 "identified", "classified", "used", "act", "acts")) or \
                      (last == "of" and prev in ("made", "composed", "consist", "consists", "formed", "built"))
        if (last in _STOPWORDS and not fine_ending) or (len(words) > 2 and words[2].lower() in _STOPWORDS
                                  and words[1].lower() in ("is", "are", "was", "were")
                                  and words[2].lower() not in ("the", "a", "an")):
            return False
        # The answer (or every distinctive word of it) must not appear in the question
        ql = q.lower()
        squash = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())  # noqa: E731 — "Displacementtime"
        if len(answer.split()) > 1 and squash(_stem(answer)) in squash(q):
            return False
        content = [_stem(w) for w in re.findall(r"[a-z0-9]+", answer.lower()) if w not in _STOPWORDS]
        if not content:
            return False
        leaked = [w for w in content if re.search(r"\b" + re.escape(w), ql)]
        return len(leaked) < len(content)

    # ── Distractor generation ─────────────────────────────────────────────────

    @staticmethod
    def _match_case(term: str, answer: str) -> str:
        words = [w if w.isupper() else w.lower() for w in term.split()]
        out = " ".join(words)
        return out[:1].upper() + out[1:] if answer[:1].isupper() else out

    def _term_distractors(self, answer: str, context: str, pos: int,
                          terms: dict[str, tuple[int, float]], span: int,
                          occ: dict[str, set[int]], a_occ: set[int], n: int = 3) -> list[str]:
        al = answer.lower()
        a_words = al.split()
        a_head = _stem(a_words[-1])
        ctx = context.lower()
        # Sentences within 3 of any place the answer is mentioned = the answer's topic
        window = {i + d for i in a_occ for d in range(-3, 4)}
        scored: list[tuple[float, str]] = []
        for t, (t_pos, quality) in terms.items():
            tl = t.lower()
            t_words = tl.split()
            if tl == al or tl in al or al in tl or _stem(tl) == _stem(al):
                continue
            if t_words[0] == a_words[0]:            # "Xylem vessel" vs "Xylem tissue": too easy
                continue
            if _term_pattern(t).search(ctx):         # also in the sentence → might be correct too
                continue
            # Mentioned alongside the answer → same topic (solvent ↔ solute, tissue ↔ xylem)
            t_occ = occ.get(t, set())
            if t_occ and a_occ:
                score = 4.0 * min(1.5, len(t_occ & window) / math.sqrt(len(a_occ) * len(t_occ)))
            else:
                score = 0.0
            score -= 5.0 * abs(t_pos - pos) / max(1, span)   # same chapter/section first
            if _stem(t_words[-1]) == a_head:
                score += 3.0                          # same kind of thing (… tissue / … tissue)
            if len(t_words) == len(a_words):
                score += 1.0
            elif abs(len(t_words) - len(a_words)) == 1:
                score += 0.5
            score += 0.5 * quality                    # defined terms > headings > lone labels
            scored.append((score, t))
        scored.sort(key=lambda x: x[0], reverse=True)

        def near_duplicate(a: str, b: str) -> bool:   # "homogeneous" vs "homogenous"
            return difflib.SequenceMatcher(None, _stem(a), _stem(b)).ratio() >= 0.85

        picked: list[str] = []
        top = [t for _, t in scored[: n * 3]]
        random.shuffle(top)
        for t in top:
            tl = t.lower()
            if near_duplicate(tl, al) or any(near_duplicate(tl, p.lower()) for p in picked):
                continue
            picked.append(self._match_case(t, answer))
            if len(picked) == n:
                break
        return picked

    def _book_terms(self, pdf_path: str) -> list[str]:
        """Key terms of the whole PDF (cached) — a fallback distractor pool for short chapters."""
        cache = self.__dict__.setdefault("_book_terms_cache", {})
        if pdf_path not in cache:
            chunks, headings, sections = self._extract(pdf_path)
            sentences, _ = self._sentences(chunks, sections)
            body = re.sub(r"\s+", " ", " ".join(chunks)).strip()
            terms = self._build_terms(headings, sentences, body)
            # defined terms first — they're real concepts, not figure labels
            cache[pdf_path] = sorted(terms, key=lambda t: -terms[t][1])
        return cache[pdf_path]

    def _book_distractors(self, pdf_path: str, answer: str, context: str,
                          have: list[str], n: int) -> list[str]:
        al, ctx = answer.lower(), context.lower()
        a_head = _stem(al.split()[-1])
        taken = [al] + [h.lower() for h in have]

        def ok(t: str) -> bool:
            tl = t.lower()
            if _term_pattern(t).search(ctx):
                return False
            return all(difflib.SequenceMatcher(None, _stem(tl), _stem(x)).ratio() < 0.85
                       and tl not in x and x not in tl for x in taken)

        pool = [t for t in self._book_terms(pdf_path) if ok(t)]
        same_kind = [t for t in pool if _stem(t.lower().split()[-1]) == a_head]
        rest = [t for t in pool[:60] if t not in same_kind]
        random.shuffle(same_kind)
        random.shuffle(rest)
        picked: list[str] = []
        for t in same_kind + rest:
            if len(picked) == n:
                break
            if all(difflib.SequenceMatcher(None, t.lower(), p.lower()).ratio() < 0.85 for p in picked):
                picked.append(self._match_case(t, answer))
        return picked

    @staticmethod
    def _number_distractors(answer: str, n: int = 3) -> list[str]:
        m = re.match(r"(\d+(?:\.\d+)?)(.*)", answer)
        if not m:
            return []
        value, unit = float(m.group(1)), m.group(2)
        dec = len(m.group(1).split(".")[1]) if "." in m.group(1) else 0
        out: list[str] = []
        for f in (2, 0.5, 1.5, 0.75, 3, 1.25, 10, 0.1):
            x = round(value * f, dec)
            if x <= 0 or x == value:
                continue
            s = (f"{x:.{dec}f}" if dec else str(int(round(x)))) + unit
            if s != answer and s not in out:
                out.append(s)
        random.shuffle(out)
        return out[:n]

    # ── Main entry point ──────────────────────────────────────────────────────

    def generate_from_pdf(
        self, pdf_path: str, num_questions: int = 9, concept_tag: str = "General",
        pages: list[int] | None = None,
    ) -> dict:
        """Generate MCQs from the PDF (or only ``pages``, 0-based — e.g. one chapter).

        With the default concept tag ("General"), each question is tagged with the
        textbook section it came from ("Plant tissues", "Animal tissues", …) so reports
        point students at a specific weak sub-topic. A teacher-given tag is kept as-is.
        """
        chunks, headings, chunk_sections = self._extract(pdf_path, pages)
        body = re.sub(r"\s+", " ", " ".join(chunks)).strip()
        if not body:
            raise ValueError("Could not extract text from PDF.")

        sentences, sentence_sections = self._sentences(chunks, chunk_sections)
        section_of = dict(zip(sentences, sentence_sections))
        auto_tag = (concept_tag or "").strip().lower() in ("", "general", "auto")
        terms = self._build_terms(headings, sentences, body)

        per_level = max(1, num_questions // 3)   # 3 for a 9-question assignment
        needed    = per_level * 3                 # always a clean multiple of 3

        # Every usable (sentence, answer) pair, in document order
        body_lower = body.lower()
        eligible: list[dict] = []
        for s in sentences:
            if not self._usable_context(s):
                continue
            picked = self._pick_answer(s, terms)
            if not picked:
                continue
            answer, kind = picked
            score = 0
            if _DEF_RE.search(s):
                score += 3
            if kind == "term":
                score += 2
            if re.search(r"\b(is|are)\b", s):
                score += 1
            if 10 <= len(s.split()) <= 30:
                score += 1
            pos = body_lower.find(s[:40].lower())
            eligible.append({"ctx": s, "answer": answer, "kind": kind, "score": score,
                             "pos": pos if pos >= 0 else 0})

        if not eligible:
            raise ValueError("No answer candidates found in PDF.")

        # Spread picks over the whole document (not just the first pages):
        # split into equal buckets and take the best unused answer from each.
        want = min(len(eligible), needed * 4)
        size = len(eligible) / want
        used: set[str] = set()
        chosen: list[dict] = []
        for b in range(want):
            bucket = eligible[int(b * size): int((b + 1) * size)] or eligible[-1:]
            for c in sorted(bucket, key=lambda c: c["score"], reverse=True):
                if _stem(c["answer"].lower()) not in used:
                    used.add(_stem(c["answer"].lower()))
                    chosen.append(c)
                    break

        # A short chapter has few distinct key terms ("friction", "static friction"…):
        # top up by asking about the same answer from a *different* sentence, ≤3× each.
        if len(chosen) < needed * 2:
            uses: dict[str, int] = {}
            for c in chosen:
                k = _stem(c["answer"].lower())
                uses[k] = uses.get(k, 0) + 1
            picked_ctx = {c["ctx"] for c in chosen}
            for c in sorted(eligible, key=lambda c: c["score"], reverse=True):
                if len(chosen) >= needed * 2:
                    break
                k = _stem(c["answer"].lower())
                if c["ctx"] in picked_ctx or uses.get(k, 0) >= 3:
                    continue
                uses[k] = uses.get(k, 0) + 1
                picked_ctx.add(c["ctx"])
                chosen.append(c)
            chosen.sort(key=lambda c: c["pos"])

        # T5 batch inference
        q_texts = self.generate_questions_batch([(c["answer"], c["ctx"]) for c in chosen])

        # Which sentences each term is mentioned in (for topic-related distractors)
        joined = " ".join(s.lower() for s in sentences)
        starts, offset = [], 0
        for s in sentences:
            starts.append(offset)
            offset += len(s) + 1

        def mentions(term: str) -> set[int]:
            return {bisect.bisect_right(starts, m.start()) - 1
                    for m in _term_pattern(term).finditer(joined)}

        occ = {t: mentions(t) for t in terms}

        span = max(1, len(body))
        questions: list[dict] = []
        seen_q: set[str] = set()
        for c, q_text in zip(chosen, q_texts):
            answer, context = c["answer"], c["ctx"]
            if not self._valid_question(q_text, answer) or q_text.lower() in seen_q:
                continue
            if c["kind"] == "number":
                distractors = self._number_distractors(answer)
            else:
                distractors = self._term_distractors(answer, context, c["pos"], terms, span,
                                                     occ, mentions(answer))
                if len(distractors) < 3 and pages is not None:
                    # Short chapter: borrow terms from the rest of the same textbook
                    distractors += self._book_distractors(pdf_path, answer, context,
                                                          distractors, 3 - len(distractors))
            i = 0
            while len(distractors) < 3:
                distractors.append(_FALLBACK_DISTRACTORS[i % len(_FALLBACK_DISTRACTORS)])
                i += 1
            options = [answer] + distractors[:3]
            random.shuffle(options)
            seen_q.add(q_text.lower())
            # Hint = the source sentence with the answer blanked out, so it nudges
            # without giving the answer away
            hint = _term_pattern(answer).sub("_____", context, count=1) if c["kind"] == "term" \
                else context.replace(answer, "_____", 1)
            questions.append({
                "questionId":             str(uuid.uuid4())[:8],
                "questionText":           q_text,
                "options":                options,
                "correctAnswer":          answer,
                "hint":                   hint[:200].strip(),
                "shortTheoryExplanation": context.strip(),
                "conceptTag":             (section_of.get(context) if auto_tag else None) or concept_tag,
                "_ctx_len":               len(context.split()),  # temp field for sorting
                "_score":                 c["score"],
            })

        # How much survived each stage — lets callers explain a low-yield chapter
        self.last_stats = {"sentences": len(sentences), "eligible": len(eligible),
                           "asked": len(chosen), "valid": len(questions)}
        if not questions:
            raise ValueError("T5 produced no usable questions from this PDF.")

        # Keep the strongest `needed`, then sort by context length (shortest = simplest =
        # Level 1, longest = hardest = Level 3) and split into 3 roughly-even levels by actual
        # count — if fewer than `needed` survived filtering, levels stay balanced (e.g. 3/2/2)
        # instead of front-loading Level 1 with a fixed per-level size.
        questions.sort(key=lambda q: q["_score"], reverse=True)
        questions = questions[:needed]
        questions.sort(key=lambda q: q["_ctx_len"])
        actual = len(questions)

        for idx, q in enumerate(questions):
            q["quizLevel"] = min(3, idx * 3 // actual + 1)
            del q["_ctx_len"]
            del q["_score"]

        bucket: dict[int, list] = {1: [], 2: [], 3: []}
        for q in questions:
            bucket[q["quizLevel"]].append(q)

        return {"level1": bucket[1], "level2": bucket[2], "level3": bucket[3]}

