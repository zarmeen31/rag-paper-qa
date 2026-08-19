"""Stage 6-9: query -> retrieval -> LLM -> answer + sources."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_google_genai import ChatGoogleGenerativeAI

from .config import CONFIG

SYSTEM_PROMPT = """You are a research assistant answering questions about a \
specific academic paper. You are given numbered excerpts from that paper.

Rules, in order of priority:
1. Answer ONLY from the excerpts. Do not use outside knowledge about the topic.
2. Cite the excerpt(s) supporting each claim inline, like [S1] or [S2][S4].
3. If the excerpts do not contain the answer, say exactly what is missing.
4. If the excerpts conflict, say so and cite both.
5. Quote numbers, metric names, and equations exactly as written.
6. Be concise. Two or three sentences unless the question needs more.
"""

USER_TEMPLATE = """Excerpts from "{title}":

{context}

Question: {question}

Answer using only the excerpts above, with [S#] citations."""


@dataclass
class Source:
    label: str
    page: int
    section: str
    text: str
    score: float = None


@dataclass
class RAGResult:
    question: str
    answer: str
    sources: list = field(default_factory=list)
    contexts: list = field(default_factory=list)


def get_llm(model=None, temperature=None):
    CONFIG.validate()
    return ChatGoogleGenerativeAI(
        model=model or CONFIG.chat_model,
        google_api_key=CONFIG.google_api_key,
        temperature=CONFIG.temperature if temperature is None else temperature,
    )


def retrieve(question, paper_id, k=None):
    from .vectorstore import get_store

    store = get_store()
    k = k or CONFIG.top_k
    where = {"paper_id": paper_id}

    if CONFIG.search_type == "mmr":
        return store.max_marginal_relevance_search(
            question, k=k, fetch_k=CONFIG.mmr_fetch_k, filter=where
        )
    return store.similarity_search(question, k=k, filter=where)


def _to_sources(docs):
    return [
        Source(
            label="S%d" % i,
            page=d.metadata.get("page", 0),
            section=d.metadata.get("section", ""),
            text=d.page_content,
        )
        for i, d in enumerate(docs, start=1)
    ]


def format_context(sources):
    return "\n\n".join(
        "[%s] (page %s, section: %s)\n%s" % (s.label, s.page, s.section, s.text)
        for s in sources
    )


def _prune_uncited(answer, sources):
    cited = set(re.findall(r"\[(S\d+)\]", answer))
    kept = [s for s in sources if s.label in cited]
    return kept or sources


def answer_question(question, paper_id, title="the paper", k=None):
    docs = retrieve(question, paper_id, k)
    if not docs:
        return RAGResult(question, "Nothing was retrieved for that question.", [], [])

    sources = _to_sources(docs)
    prompt = USER_TEMPLATE.format(
        title=title, context=format_context(sources), question=question
    )
    llm = get_llm()
    response = llm.invoke([("system", SYSTEM_PROMPT), ("human", prompt)])
    text = response.content if isinstance(response.content, str) else str(response.content)

    return RAGResult(
        question=question,
        answer=text.strip(),
        sources=_prune_uncited(text, sources),
        contexts=[s.text for s in sources],
    )


def stream_answer(question, paper_id, title="the paper", k=None):
    docs = retrieve(question, paper_id, k)
    if not docs:
        yield RAGResult(question, "Nothing was retrieved for that question.", [], [])
        return

    sources = _to_sources(docs)
    prompt = USER_TEMPLATE.format(
        title=title, context=format_context(sources), question=question
    )
    llm = get_llm()

    buffer = []
    for chunk in llm.stream([("system", SYSTEM_PROMPT), ("human", prompt)]):
        piece = chunk.content if isinstance(chunk.content, str) else ""
        if piece:
            buffer.append(piece)
            yield piece

    full = "".join(buffer).strip()
    yield RAGResult(
        question=question,
        answer=full,
        sources=_prune_uncited(full, sources),
        contexts=[s.text for s in sources],
    )
