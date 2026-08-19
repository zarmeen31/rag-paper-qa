"""Streamlit front end for the RAG Research Paper Q&A Assistant."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import CONFIG
from src.evaluation import EvalCase, aggregate, evaluate_case
from src.ingest import build_chunks, ingestion_report, paper_id_for
from src.rag import RAGResult, stream_answer
from src.vectorstore import add_documents, delete_paper, list_papers, paper_exists

st.set_page_config(page_title="Paper Q&A", page_icon="📄", layout="wide")


# ---------------------------------------------------------------------------
# Sidebar: library management
# ---------------------------------------------------------------------------

def sidebar() -> dict | None:
    st.sidebar.title("📄 Paper library")

    try:
        CONFIG.validate()
    except RuntimeError as exc:
        st.sidebar.error(str(exc))
        st.stop()

    uploads = st.sidebar.file_uploader(
        "Add papers (PDF)", type="pdf", accept_multiple_files=True
    )

    if uploads and st.sidebar.button("Index papers", type="primary"):
        for up in uploads:
            path = CONFIG.upload_dir / up.name
            path.write_bytes(up.getbuffer())
            pid = paper_id_for(path)

            if paper_exists(pid):
                st.sidebar.info(f"{up.name} already indexed — skipped.")
                continue

            with st.sidebar.status(f"Processing {up.name}", expanded=False) as status:
                try:
                    docs = build_chunks(path)
                    report = ingestion_report(docs)
                    status.write(
                        f"{report['chunks']} chunks over {report['pages']} pages "
                        f"(avg {report['avg_chunk_chars']} chars)"
                    )
                    add_documents(docs)
                    status.update(label=f"Indexed {up.name}", state="complete")
                    st.session_state[f"report::{pid}"] = report
                except Exception as exc:
                    status.update(label=f"Failed: {up.name}", state="error")
                    st.sidebar.error(str(exc))
        st.rerun()

    papers = list_papers()
    if not papers:
        st.sidebar.caption("No papers indexed yet.")
        return None

    labels = {p["title"]: p for p in papers}
    choice = st.sidebar.selectbox("Active paper", list(labels))
    selected = labels[choice]
    st.sidebar.caption(f"{selected['chunks']} chunks · id `{selected['paper_id']}`")

    with st.sidebar.expander("Retrieval settings"):
        CONFIG.top_k = st.slider("Chunks retrieved (k)", 2, 12, CONFIG.top_k)
        CONFIG.search_type = st.radio(
            "Search", ["mmr", "similarity"],
            index=0 if CONFIG.search_type == "mmr" else 1,
            horizontal=True,
        )

    if st.sidebar.button("Remove this paper"):
        delete_paper(selected["paper_id"])
        st.session_state.pop("messages", None)
        st.rerun()

    return selected


# ---------------------------------------------------------------------------
# Chat tab
# ---------------------------------------------------------------------------

def render_sources(result: RAGResult) -> None:
    with st.expander(f"Sources ({len(result.sources)})"):
        for s in result.sources:
            st.markdown(f"**[{s.label}]** page {s.page} · *{s.section}*")
            st.caption(s.text[:700] + ("…" if len(s.text) > 700 else ""))
            st.divider()


def chat_tab(paper: dict) -> None:
    st.session_state.setdefault("messages", [])

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("result"):
                render_sources(msg["result"])

    question = st.chat_input(f"Ask about {paper['title']}")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        buffer, final = "", None
        for piece in stream_answer(question, paper["paper_id"], paper["title"]):
            if isinstance(piece, RAGResult):
                final = piece
            else:
                buffer += piece
                placeholder.markdown(buffer + "▌")
        placeholder.markdown(buffer or (final.answer if final else ""))
        if final:
            render_sources(final)

    st.session_state.messages.append(
        {"role": "assistant", "content": buffer, "result": final}
    )


# ---------------------------------------------------------------------------
# Evaluation tab
# ---------------------------------------------------------------------------

def eval_tab(paper: dict) -> None:
    st.subheader("RAG evaluation")
    st.caption(
        "Faithfulness = not hallucinating · Relevance = answers the question · "
        "Context precision = retrieval isn't noisy · Context recall = retrieval "
        "isn't missing things."
    )

    default = Path("eval/eval_set.example.json")
    uploaded = st.file_uploader("Evaluation set (JSON)", type="json")

    if uploaded:
        raw = json.loads(uploaded.getvalue())
    elif default.exists():
        raw = json.loads(default.read_text())
        st.info("Using the example set — replace it with questions about your paper.")
    else:
        st.warning("No evaluation set found.")
        return

    cases_raw = raw["cases"] if isinstance(raw, dict) else raw
    st.write(f"{len(cases_raw)} cases loaded.")

    if not st.button("Run evaluation", type="primary"):
        return

    cases = [
        EvalCase(
            question=c["question"],
            reference_answer=c.get("reference_answer", ""),
            reference_context=c.get("reference_context", ""),
            tags=c.get("tags", []),
        )
        for c in cases_raw
    ]

    results, bar = [], st.progress(0.0)
    log = st.empty()
    for i, case in enumerate(cases, 1):
        log.caption(f"[{i}/{len(cases)}] {case.question[:80]}")
        try:
            results.append(evaluate_case(case, paper["paper_id"], paper["title"]))
        except Exception as exc:
            st.warning(f"Case failed: {case.question[:50]} — {exc}")
        bar.progress(i / len(cases))
        time.sleep(0.5)
    log.empty()

    if not results:
        st.error("No cases completed.")
        return

    summary = aggregate(results)
    cols = st.columns(5)
    for col, key in zip(
        cols,
        ["faithfulness", "answer_relevance", "context_precision",
         "context_recall", "answer_correctness"],
    ):
        val = summary.get(key)
        col.metric(key.replace("_", " ").title(), "—" if val is None else f"{val:.2f}")

    df = pd.DataFrame([r.as_dict() for r in results])
    st.dataframe(df, use_container_width=True)
    st.download_button(
        "Download results CSV",
        df.to_csv(index=False),
        file_name="rag_eval_results.csv",
        mime="text/csv",
    )

    problems = [r for r in results if r.unsupported_claims or r.missing_claims]
    if problems:
        st.subheader("Cases worth inspecting")
        for r in problems:
            with st.expander(r.question[:90]):
                if r.unsupported_claims:
                    st.error("Unsupported claims (hallucination):")
                    for c in r.unsupported_claims:
                        st.write(f"- {c}")
                if r.missing_claims:
                    st.warning("Reference facts not retrieved (retrieval gap):")
                    for c in r.missing_claims:
                        st.write(f"- {c}")


# ---------------------------------------------------------------------------

def main() -> None:
    st.title("Research Paper Q&A Assistant")
    paper = sidebar()

    if not paper:
        st.info("Upload a paper in the sidebar to get started.")
        st.markdown(
            "**Pipeline:** PDF → clean & chunk → embed → Chroma → "
            "retrieve top-k → grounded answer with `[S#]` citations → evaluate."
        )
        return

    chat, evaluation = st.tabs(["Ask", "Evaluate"])
    with chat:
        chat_tab(paper)
    with evaluation:
        eval_tab(paper)


if __name__ == "__main__":
    main()
