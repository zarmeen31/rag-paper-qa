# RAG Research Paper Q&A Assistant

Ask questions about a research paper, get answers grounded in the paper with
page-level citations — plus an evaluation harness that measures whether the
retrieval is actually working.

The interesting part of this project is not the chatbot. It is the measurement.

---

## The problem this project is really about

A RAG system that is completely broken still produces confident, fluent,
plausible answers. You cannot tell by reading them.

Early on, this system was asked:

> "What benefit and drawback percentages did each participant group report?"

It returned a fluent answer and five sources. None of them contained the table
with the actual numbers. It had retrieved the sentence that *introduces* the
table — "the distribution of the answers ... is shown in Table 1" — and missed
the table itself.

That failure is what the rest of this project measures.

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
cp .env.example .env            # add your GOOGLE_API_KEY
streamlit run app.py
```

Get a free key at https://aistudio.google.com/apikey

---

## Architecture

| Stage | File |
|---|---|
| PDF → clean text → chunks | `src/ingest.py` |
| Embeddings + vector store | `src/vectorstore.py` |
| Query → retrieval → LLM → answer + sources | `src/rag.py` |
| LLM-judge evaluation | `src/evaluation.py`, `eval/run_eval.py` |
| Retrieval-only evaluation | `eval/retrieval_eval.py` |
| Streamlit UI | `app.py` |

Stack: Streamlit, Google Gemini, Chroma, LangChain, pdfplumber.

**Ingestion is where most RAG systems actually fail.** `ingest.py` rejoins
hyphenated line breaks, strips running headers and page numbers, normalises
ligatures, preserves section headings as their own blocks, and truncates the
reference list (bibliographies are keyword-dense and answer-free, so they
crowd out real content at retrieval time).

---

## Evaluation

Two evaluators, because the first one hit a wall.

**`eval/run_eval.py`** — LLM-as-judge. Scores faithfulness, answer relevance,
context precision, context recall, and answer correctness using claim
decomposition rather than a holistic "rate this 1-10" call, because holistic
LLM scores are close to noise.

Cost: ~6 API calls per case. On the Gemini free tier that is 20 requests per
day per model, so a 17-case run dies about a third of the way through.

**`eval/retrieval_eval.py`** — retrieval-only. No generation, no judge, no
quota limit.

The insight: if a fact is not in the retrieved chunks, no downstream LLM can
produce it. That is a retrieval failure, and it is measurable with string
matching alone. The evaluator extracts salient terms (numbers, proper nouns,
distinctive content words) from a hand-written reference answer and checks
whether they appear in the retrieved context. Questions whose correct answer
is "the paper does not say this" are excluded from recall rather than scored
zero.

Metrics: `fact_recall`, `numeric_recall`, `mrr`, `hit_at_1`, `hit_at_3`,
`miss_rate` — reported overall and broken down by question type.

```bash
python eval/retrieval_eval.py --set eval/my_paper.json --k 10 --search similarity
```

---

## Results

Test document: Gocen & Aydemir (2020), *Artificial Intelligence in Education
and Schools*. 51 chunks. 17 hand-written questions with reference answers,
including 4 deliberately unanswerable ones (without those, faithfulness scores
near 1.0 even on a system that hallucinates freely).

Fact recall by question type, across four retrieval configurations:

| question type | k=5 mmr | k=10 mmr | k=10 similarity | k=20 mmr |
|---|---|---|---|---|
| table | 0.506 | 0.833 | **1.000** | 1.000 |
| buried-detail | 0.537 | 0.799 | 0.799 | 0.821 |
| findings | 0.600 | 1.000 | 1.000 | 1.000 |
| numeric | 0.685 | 0.882 | **1.000** | 1.000 |
| multi-hop | 0.682 | 1.000 | 1.000 | 1.000 |
| factual | 0.698 | 0.988 | 0.988 | 0.988 |
| method | 0.838 | 0.977 | 1.000 | 1.000 |
| citation | 1.000 | 1.000 | 1.000 | 1.000 |

### Finding 1 — the failure was coverage, not ranking

MRR was 1.0 in every category, in every configuration. Whenever a relevant
chunk appeared in the results at all, it was ranked first. The retriever knew
where to look; it just wasn't looking at enough. That rules out reranking as a
fix and points at retrieval depth.

### Finding 2 — MMR underperformed plain similarity on this document

MMR was the original default, chosen because papers restate themselves and
near-duplicate chunks seemed likely to crowd out the answer. The data
contradicted that.

At k=10, switching from MMR to plain similarity took table recall from 0.833 to
1.000 and numeric recall from 0.882 to 1.000, at no additional cost. Plausible
mechanism: MMR's diversity penalty demotes chunks that resemble one another —
which is exactly what the rows of a single table do.

### Finding 3 — k=10 is the plateau

k=20 gained +0.02 on one category and nothing anywhere else. Doubling
retrieval depth beyond 10 is wasted compute on this document.

**Recommended configuration: `TOP_K=10`, `SEARCH_TYPE=similarity`.**

---

## Open problem

Buried-detail questions sit at ~0.80 recall and do not improve with more
chunks or a different search type. These are facts like the funding programme
in the acknowledgements, or a single investment figure cited once in the
introduction. They live in sections that are semantically unrelated to how
anyone would naturally phrase the question, so dense retrieval alone cannot
reach them reliably.

A related observation: retrieval quality depends heavily on question wording.
The same table was missed entirely by "what benefit and drawback percentages
did each group report?" and ranked first by "what percentage did expert
engineers give for benefit?" — same index, same settings. Generic category
words that recur throughout a document retrieve worse than rare, specific
terms.

Next steps would be hybrid retrieval (BM25 alongside dense vectors) and query
rewriting, both aimed at the same weakness.

---

## Known limits

- Scanned PDFs have no text layer; run OCR first.
- Tables and equations flatten in any text extractor. Numeric questions are
  where this fails first, which is why the eval set is weighted toward them.
- Retrieval is filtered to one paper at a time by design.
- Results are from a single document. Whether Finding 2 generalises to papers
  with different structure is untested.