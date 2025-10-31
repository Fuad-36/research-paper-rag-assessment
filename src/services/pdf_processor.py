import fitz  # pymupdf
import re
import logging
from typing import List, Dict, Any, Optional,Callable
from src.config import settings
import nltk

# set up module logger (the application can configure handlers/format)
logger = logging.getLogger(__name__)

# Ensure punkt tokenizer is available — try to find, otherwise attempt download
for resource in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        try:
            nltk.download(resource)
            logger.info("Downloaded NLTK tokenizer: %s", resource)
        except Exception as e:
            logger.warning("Failed to download NLTK tokenizer '%s': %s", resource, e)

from nltk.tokenize import sent_tokenize  # may still raise if not available


SECTION_HEADERS = [
    r"abstract",
    r"introduction",
    r"background",
    r"related work",
    r"method",
    r"methodology",
    r"experimental setup",
    r"experiments",
    r"results",
    r"discussion",
    r"conclusion",
    r"conclusions",
    r"references",
]


def _detect_section(text: str) -> str:
    """
    Heuristic section detection: check top lines first, then fallback to searching whole page.
    Returns capitalized header name or 'Body'.
    """
    try:
        lines = text.splitlines()
    except Exception:
        return "Body"

    # check first few lines for a header
    for line in lines[:6]:
        for h in SECTION_HEADERS:
            if re.search(rf"^\s*{h}\b", line, re.IGNORECASE):
                return h.capitalize()

    # fallback: search anywhere in the page
    for h in SECTION_HEADERS:
        if re.search(rf"\b{h}\b", text, re.IGNORECASE):
            return h.capitalize()

    return "Body"


def _largest_font_text_on_page(page) -> str:
    """
    Return the best candidate title by examining spans grouped by font-size.
    Rule: choose the largest font-size group whose combined text has > 3 words.
    If no such group exists, fall back to the largest group or the longest span.
    """
    try:
        d = page.get_text("dict")
        spans = []  # preserve order: list of (size, text)
        for block in d.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    txt = span.get("text", "").strip()
                    if not txt or len(txt) < 2:
                        continue
                    size = float(span.get("size", 0) or 0)
                    spans.append((size, txt))

        if not spans:
            return ""

        # Collect unique sizes in descending order (allow rounding diffs)
        sizes = sorted({round(s, 1) for s, _ in spans}, reverse=True)

        # helper to build combined text for a given target size (allow small tolerance)
        def combined_for_size(target_size):
            pieces = []
            for s, t in spans:
                if abs(s - target_size) < 0.5:
                    pieces.append(t)
            joined = " ".join(pieces)
            joined = re.sub(r'\s{2,}', ' ', joined).strip()
            return joined

        # try sizes in descending order until find >= 4 words
        for s in sizes:
            candidate = combined_for_size(s)
            if len(candidate.split()) > 3:
                title = candidate
                break
        else:
            # fallback: largest size combined, even if short
            title = combined_for_size(sizes[0])

        # Cleanup common junk prefixes/suffixes
        title = re.sub(r'^(ARTICLE|REVIEW|COMMENTARY|PAPER|Original Article|Received:)\s+', '', title, flags=re.I)
        title = re.sub(r'\s+DOI[:\s].*$', '', title, flags=re.I)
        title = re.sub(r'\s{2,}', ' ', title).strip()

        # final short fallback to the longest single span
        if len(title) < 5:
            longest_span = max((t for _, t in spans), key=len)
            title = longest_span.strip()

        return title
    except Exception:
        return ""


def _first_nonempty_lines(text: str, n: int = 10) -> List[str]:
    return [l.strip() for l in text.splitlines() if l.strip()][:n]

def _clean_name_token(tok: str) -> str:
    # remove footnote markers and stray punctuation
    t = re.sub(r'[\d\*\)\(\[\]]+', '', tok)
    t = re.sub(r'\s{2,}', ' ', t)
    t = t.strip(' ,;:.')
    return t

import re

def extract_authors_after_title(text: str, title: str, max_chars: int = 1200) -> str:
    """
    1) Try to extract ONLY names that are followed by a superscript/digit/footnote marker
       (this is the mandatory form you specified). If any such names are found, return them
       in order as a comma-separated list.
    2) If NO superscript-marked names are present, fall back to the rule:
       - find 'and' and take the next two words as the final author,
       - extract preceding capitalized name-like groups (at least two tokens),
       - stop scanning after this (do not iterate further).
    Returns: comma-separated author names (or empty string).
    """
    if not text:
        return ""

    # Normalize whitespace for stable searching
    norm_text = re.sub(r'\s+', ' ', text).strip()

    # Find start after title (robust partial match if necessary)
    start_idx = 0
    if title:
        norm_title = re.sub(r'\s+', ' ', title).strip()
        m = re.search(re.escape(norm_title), norm_text, re.IGNORECASE)
        if m:
            start_idx = m.end()
        else:
            # try partial (first up to 10 words)
            first_words = " ".join(norm_title.split()[:10])
            if first_words:
                m2 = re.search(re.escape(first_words), norm_text, re.IGNORECASE)
                if m2:
                    start_idx = m2.end()

    # Slice a reasonable region after the title
    after = norm_text[start_idx:start_idx + max_chars] if start_idx < len(norm_text) else norm_text[:max_chars]

    # Cut at common section delimiters so we don't extract affiliations, references, etc.
    stop = re.search(r'\b(Abstract|Received|Correspondence|Affiliations|Keywords|Introduction|References|Acknowledg)', after, re.I)
    if stop:
        after = after[:stop.start()]

    # ---- 1) PRIMARY PATH: require superscripts/digits after each name ----
    # Name token accepts initials and diacritics/punct: e.g., "M.", "M. I.", "Marija", "Dražen Žgaljić"
    name_token = r'[A-Z][A-Za-z\.\'`´’\-À-ÖØ-öø-ÿ]+'  # allow accented letters
    # Require at least two tokens per name and then require a superscript/digit/footnote marker
    superscript_marker = r'(?:[\u00B9\u00B2\u00B3\u2070-\u2079]|\d+|[\*\†\‡])'
    name_with_marker_re = re.compile(
        rf'({name_token}(?:\s+{name_token})+)\s*{superscript_marker}',
        flags=re.UNICODE
    )

    supers_matches = [m.group(1).strip() for m in name_with_marker_re.finditer(after)]

    def clean_name(nm: str) -> str:
        # Remove any leftover digits, superscripts, or footnote symbols and trim punctuation
        nm = re.sub(r'[\u00B9\u00B2\u00B3\u2070-\u2079]|\d+|[\*\†\‡]', '', nm)
        nm = re.sub(r'^[,;:\-\s]+|[,;:\-\s]+$', '', nm).strip()
        # remove trailing affiliation words if accidentally captured
        nm = re.sub(r'\b(University|Institute|Department|Faculty|School|Laboratory|College|Center|Centre|Please cite|DOI|Abstract)\b.*$', '', nm, flags=re.I).strip()
        return nm

    cleaned = []
    seen = set()
    # If we found superscript-marked names, return them (mandatory path)
    if supers_matches:
        for s in supers_matches:
            c = clean_name(s)
            if not c:
                continue
            key = re.sub(r'\s+', ' ', c).lower()
            if key not in seen:
                seen.add(key)
                # ensure it looks like "Firstname Lastname" (>=2 tokens)
                if len(c.split()) >= 2:
                    cleaned.append(c)
        return ", ".join(cleaned) if cleaned else ""

    # ---- 2) FALLBACK PATH: NO superscripts present -> use `and` + next two words (stop after this) ----
    m_and = re.search(r'\band\b', after, re.I)
    if not m_and:
        # No 'and' either, try looser capitalization-based capture in the slice
        # Extract capitalized sequences of at least two tokens (best-effort)
        loose_re = re.compile(r'([A-Z][A-Za-z\.\'`´’\-À-ÖØ-öø-ÿ]+(?:\s+[A-Z][A-Za-z\.\'`´’\-À-ÖØ-öø-ÿ]+)+)')
        for m in loose_re.finditer(after):
            c = clean_name(m.group(1))
            if c and len(c.split()) >= 2:
                key = re.sub(r'\s+', ' ', c).lower()
                if key not in seen:
                    seen.add(key)
                    cleaned.append(c)
        return ", ".join(cleaned) if cleaned else ""

    # If 'and' found, take left part (before 'and') and right two words (after 'and')
    before_and = after[:m_and.start()].strip()
    after_and = after[m_and.end():].strip()

    # Capture the next two words after 'and' as the final author (require both are capitalized tokens)
    m_next = re.match(r'\s*([A-Z][A-Za-z\.\'`´’\-À-ÖØ-öø-ÿ]+)\s+([A-Z][A-Za-z\.\'`´’\-À-ÖØ-öø-ÿ]+)', after_and)
    last_author = ""
    if m_next:
        last_author = f"{m_next.group(1)} {m_next.group(2)}"

    # From the before_and segment, extract capitalized name-like groups (>=2 tokens)
    name_re_no_marker = re.compile(r'([A-Z][A-Za-z\.\'`´’\-À-ÖØ-öø-ÿ]+(?:\s+[A-Z][A-Za-z\.\'`´’\-À-ÖØ-öø-ÿ]+)+)')
    left_matches = [m.group(1).strip() for m in name_re_no_marker.finditer(before_and)]

    for lm in left_matches:
        c = clean_name(lm)
        if c and len(c.split()) >= 2:
            key = re.sub(r'\s+', ' ', c).lower()
            if key not in seen:
                seen.add(key)
                cleaned.append(c)

    # add last_author if valid
    if last_author:
        c = clean_name(last_author)
        if c and len(c.split()) >= 2:
            key = re.sub(r'\s+', ' ', c).lower()
            if key not in seen:
                seen.add(key)
                cleaned.append(c)

    return ", ".join(cleaned) if cleaned else ""




def extract_pdf(file_path: str) -> Dict[str, Any]:
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        logger.error("Failed to open PDF '%s': %s", file_path, e)
        raise ValueError(f"Failed to open PDF '{file_path}': {e}")

    pages: List[Dict[str, Any]] = []
    for i, page in enumerate(doc):
        try:
            text = page.get_text("text") or ""
        except Exception as e:
            logger.warning("Failed to extract text from page %d of %s: %s", i + 1, file_path, e)
            text = ""
        pages.append({"page_num": i + 1, "text": text})
    try:
        doc.close()
    except Exception:
        pass

    num_pages = len(pages)
    first = pages[0]["text"] if pages else ""
    title = ""
    authors = ""
    year: Optional[int] = None
    debug = {"first_lines": _first_nonempty_lines(first, 20)}

    # title heuristics 
    try:
        doc2 = fitz.open(file_path)
        if doc2.page_count > 0:
            page0 = doc2.load_page(0)
            candidate = _largest_font_text_on_page(page0)
            if candidate and len(candidate) > 3:
                title = candidate[:1000]
        doc2.close()
    except Exception:
        logger.debug("largest-font title heuristic failed; falling back to line-based.")

    if not title:
        try:
            lines = [l.strip() for l in first.splitlines() if l.strip()]
            for ln in lines:
                if re.match(r"^(Citation:|©|Energies|Joule|Received:|Published:)", ln, re.I):
                    continue
                if len(ln) < 5:
                    continue
                title = ln[:250]
                break
            if not title and lines:
                title = lines[0][:250]
        except Exception:
            logger.debug("Fallback title heuristic failed.")

    # ----- NEW authors logic (your requested heuristic) -----
    try:
        authors = extract_authors_after_title(first, title)
        # small fallback if not found: try combining first two non-empty lines after title
        if not authors:
            lines = [l.strip() for l in first.splitlines() if l.strip()]
            if title in lines:
                ti = lines.index(title)
                cand = []
                for j in range(ti+1, min(ti+5, len(lines))):
                    cand.append(re.sub(r'[\d\*\)\(]+', '', lines[j]).strip())
                if cand:
                    # join and then try to clean by cutting at 'University'/'Please cite'
                    joined = " ".join(cand)
                    authors = extract_authors_after_title(joined, "")
    except Exception:
        logger.debug("Authors detection failed; leaving blank.")
    # -------------------------------------------------------

    # year detection
    try:
        joint = "\n".join(p["text"] for p in pages[:3])
        m = re.search(r"\b(19|20)\d{2}\b", joint)
        if not m:
            joint = "\n".join(p["text"] for p in pages)
            m = re.search(r"\b(19|20)\d{2}\b", joint)
        if m:
            year = int(m.group(0))
    except Exception:
        logger.debug("Year detection failed.")

    return {
        "title": title,
        "authors": authors,
        "year": year,
        "pages": pages,
        "num_pages": num_pages,
    }



def _naive_sentence_split(text: str) -> List[str]:
    """
    Fallback sentence splitter if NLTK's sent_tokenize isn't available.
    Very crude: split on dot/question/exclamation followed by space and capital letter or EOL.
    """
    # keep it simple and conservative
    pieces = re.split(r"(?<=[\.\?\!])\s+", text.strip())
    return [p for p in pieces if p]


def chunk_text_with_sections(
    pages: List[Dict[str, Any]],
    max_chars: int = settings.MAX_CHUNK_CHARS,
    overlap: int = settings.CHUNK_OVERLAP_CHARS,
) -> List[Dict[str, Any]]:
    """
    Convert list of pages (each {'page_num': int, 'text': str}) into section-aware chunks.
    Returns list of chunks: {'text', 'page_start', 'page_end', 'section'}.
    """

    # Validate inputs and normalize
    if not pages or not isinstance(pages, list):
        logger.info("No pages provided to chunk_text_with_sections; returning empty list.")
        return []

    try:
        # ensure sensible numeric parameters
        max_chars = int(max_chars)
        overlap = int(overlap)
        if max_chars <= 0:
            logger.warning("Invalid max_chars (%s) – falling back to 1800.", max_chars)
            max_chars = 1800
        if overlap < 0:
            overlap = 0
    except Exception:
        logger.warning("Invalid chunk parameters; using defaults.")
        max_chars = 1800
        overlap = 200

    chunks: List[Dict[str, Any]] = []
    current_text = ""
    current_start: Optional[int] = None
    current_section: Optional[str] = None

    try:
        for p in pages:
            # defensive: ensure shape
            page_num = p.get("page_num") if isinstance(p, dict) else None
            page_text = (p.get("text") if isinstance(p, dict) else "") or ""
            page_text = page_text.strip()
            if not page_text:
                continue

            section = _detect_section(page_text)

            # split page into paragraphs -> then sentences to control chunk size
            paragraphs = [para.strip() for para in page_text.split("\n\n") if para.strip()]
            for para in paragraphs:
                if not current_text:
                    current_start = page_num or 1
                    current_section = section or "Body"

                # if adding this paragraph would exceed size and we have an existing chunk -> emit
                if len(current_text) + len(para) > max_chars:
                    # flush current chunk if it has content
                    if current_text.strip():
                        chunks.append(
                            {
                                "text": current_text.strip(),
                                "page_start": current_start or (page_num or 1),
                                "page_end": page_num or current_start or 1,
                                "section": current_section or "Body",
                            }
                        )

                    # start new chunk with overlap (last N chars)
                    overlap_text = current_text[-overlap:] if overlap < len(current_text) else current_text
                    # ensure overlap_text and paragraph are separated
                    if overlap_text:
                        current_text = overlap_text + "\n" + para
                    else:
                        current_text = para
                    current_start = page_num or current_start
                    current_section = section or current_section
                else:
                    # normal append
                    if current_text:
                        current_text += "\n\n" + para
                    else:
                        current_text = para

        # push remaining text
        if current_text and current_text.strip():
            chunks.append(
                {
                    "text": current_text.strip(),
                    "page_start": current_start or (pages[0].get("page_num") if pages else 1),
                    "page_end": pages[-1].get("page_num") if pages else current_start or 1,
                    "section": current_section or "Body",
                }
            )

    except Exception as e:
        logger.exception("Error while forming chunks: %s", e)
        # fail-safe: return what we have so far
        if not chunks:
            return []
        # continue to postprocessing with partial chunks

    # Postprocess: ensure chunks are not overly large by splitting at sentence boundaries
    final: List[Dict[str, Any]] = []
    for c in chunks:
        text = c.get("text", "")
        if len(text) <= max_chars:
            final.append(c)
            continue

        # prefer NLTK sent_tokenize; if it fails, fallback to naive splitter
        try:
            sents = sent_tokenize(text)
            if not sents:
                sents = _naive_sentence_split(text)
        except Exception as e:
            logger.warning("sent_tokenize failed, using naive splitter: %s", e)
            sents = _naive_sentence_split(text)

        cur = ""
        cur_start = c.get("page_start")
        for s in sents:
            # account for leading space when concatenating
            if not cur:
                candidate = s
            else:
                candidate = cur + " " + s

            if len(candidate) > max_chars:
                # if cur is empty and single sentence > max_chars, we still include it (cannot split further)
                if cur.strip():
                    final.append(
                        {
                            "text": cur.strip(),
                            "page_start": cur_start,
                            "page_end": c.get("page_end"),
                            "section": c.get("section"),
                        }
                    )
                # start new current with this sentence (even if big)
                cur = s
                cur_start = c.get("page_start")
            else:
                cur = candidate

        if cur.strip():
            final.append(
                {
                    "text": cur.strip(),
                    "page_start": cur_start,
                    "page_end": c.get("page_end"),
                    "section": c.get("section"),
                }
            )

    return final
