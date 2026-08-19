"""Stage 1-3: PDF -> clean text -> chunks."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .config import CONFIG

SECTION_PATTERNS = [
    r"^\s*(?:\d+\.?\d*\s+)?(abstract)\s*$",
    r"^\s*(?:\d+\.?\d*\s+)?(introduction)\s*$",
    r"^\s*(?:\d+\.?\d*\s+)?(related work|background|literature review)\s*$",
    r"^\s*(?:\d+\.?\d*\s+)?(method(?:s|ology)?|approach|model|proposed method)\s*$",
    r"^\s*(?:\d+\.?\d*\s+)?(experiment(?:s|al setup)?|setup|dataset(?:s)?)\s*$",
    r"^\s*(?:\d+\.?\d*\s+)?(result(?:s)?|evaluation|findings)\s*$",
    r"^\s*(?:\d+\.?\d*\s+)?(discussion|analysis|ablation(?: study)?)\s*$",
    r"^\s*(?:\d+\.?\d*\s+)?(conclusion(?:s)?|future work)\s*$",
    r"^\s*(?:\d+\.?\d*\s+)?(limitations)\s*$",
    r"^\s*(?:\d+\.?\d*\s+)?(acknowledg(?:e)?ments?)\s*$",
    r"^\s*(?:\d+\.?\d*\s+)?(references|bibliography)\s*$",
]
_SECTION_RE = re.compile("|".join(SECTION_PATTERNS), re.IGNORECASE | re.MULTILINE)
_NUMBERED_HEADING_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+([A-Z][^\n]{2,70})\s*$")


@dataclass
class Page:
    number: int
    text: str


def extract_pages(pdf_path):
    pdf_path = Path(pdf_path)
    pages = []
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text(x_tolerance=1.5, y_tolerance=2) or ""
                pages.append(Page(number=i, text=text))
        if any(p.text.strip() for p in pages):
            return pages
    except Exception:
        pages = []

    from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    for i, page in enumerate(reader.pages, start=1):
        pages.append(Page(number=i, text=page.extract_text() or ""))

    if not any(p.text.strip() for p in pages):
        raise ValueError(
            "No extractable text in %s. It is probably a scanned image."
            % pdf_path.name
        )
    return pages


_LIGATURES = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi",
    "\ufb04": "ffl", "\u2013": "-", "\u2014": "-", "\u2018": "'",
    "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u00a0": " ",
}


def _normalise_glyphs(text):
    for bad, good in _LIGATURES.items():
        text = text.replace(bad, good)
    return text


def _find_repeated_lines(pages, threshold=0.4):
    if len(pages) < 4:
        return set()
    counts = Counter()
    for page in pages:
        seen = set()
        for line in page.text.splitlines():
            line = line.strip()
            if 3 < len(line) < 120:
                seen.add(line)
        counts.update(seen)
    cutoff = max(2, int(len(pages) * threshold))
    return {line for line, n in counts.items() if n >= cutoff}


def _strip_page_numbers(line):
    s = line.strip()
    if re.fullmatch(r"\d{1,4}", s):
        return True
    if re.fullmatch(r"(page\s*)?\d{1,4}\s*(of|/)\s*\d{1,4}", s, re.I):
        return True
    if re.match(r"^arxiv:\s*\d+", s, re.I):
        return True
    return False


def clean_pages(pages):
    boilerplate = _find_repeated_lines(pages)
    cleaned = []

    for page in pages:
        lines = []
        for line in page.text.splitlines():
            s = line.strip()
            if not s:
                lines.append("")
                continue
            if s in boilerplate or _strip_page_numbers(s):
                continue
            lines.append(line)

        protected = []
        for line in lines:
            s = line.strip()
            if s and (_SECTION_RE.match(s) or _NUMBERED_HEADING_RE.match(s)):
                protected.extend(["", s, ""])
            else:
                protected.append(line)
        text = "\n".join(protected)

        text = _normalise_glyphs(text)
        text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
        text = re.sub(r"(?<![.:;?!\n])\n(?!\n)", " ", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        cleaned.append(Page(number=page.number, text=text.strip()))

    return cleaned


def truncate_at_references(pages):
    out = []
    for page in pages:
        match = None
        for m in re.finditer(
            r"^\s*(?:\d+\.?\s*)?(references|bibliography)\s*$",
            page.text, re.IGNORECASE | re.MULTILINE,
        ):
            match = m
        if match and match.start() > 0.15 * max(len(page.text), 1):
            head = page.text[: match.start()].strip()
            if head:
                out.append(Page(number=page.number, text=head))
            break
        if match and match.start() <= 0.15 * max(len(page.text), 1):
            break
        out.append(page)
    return out or pages


def _section_for(text, current):
    for line in text.splitlines()[:4]:
        m = _SECTION_RE.match(line.strip())
        if m:
            label = next(g for g in m.groups() if g)
            return label.title()
        m2 = _NUMBERED_HEADING_RE.match(line.strip())
        if m2:
            return m2.group(2).strip().title()
    return current


def paper_id_for(path):
    h = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
    return "%s-%s" % (Path(path).stem[:40], h)


def build_chunks(pdf_path, title=None):
    pdf_path = Path(pdf_path)
    pages = extract_pages(pdf_path)
    if CONFIG.drop_references:
        pages = truncate_at_references(pages)
    pages = clean_pages(pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG.chunk_size,
        chunk_overlap=CONFIG.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    pid = paper_id_for(pdf_path)
    docs = []
    current_section = "Front Matter"
    index = 0

    for page in pages:
        if not page.text.strip():
            continue
        for piece in splitter.split_text(page.text):
            piece = piece.strip()
            if len(piece) < CONFIG.min_chunk_chars:
                continue
            current_section = _section_for(piece, current_section)
            docs.append(Document(
                page_content=piece,
                metadata={
                    "paper_id": pid,
                    "title": title or pdf_path.stem,
                    "source": pdf_path.name,
                    "page": page.number,
                    "section": current_section,
                    "chunk_index": index,
                },
            ))
            index += 1

    if not docs:
        raise ValueError("%s produced no usable chunks." % pdf_path.name)
    return docs


def ingestion_report(docs):
    sections = Counter(d.metadata["section"] for d in docs)
    lengths = [len(d.page_content) for d in docs]
    return {
        "chunks": len(docs),
        "pages": len({d.metadata["page"] for d in docs}),
        "avg_chunk_chars": round(sum(lengths) / len(lengths)),
        "sections": dict(sections),
    }
