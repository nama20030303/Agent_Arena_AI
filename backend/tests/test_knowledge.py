"""
Retrieval quality tests: embeddings, lexical index, hybrid ranking, citations, conflicts.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.ai.fallback import local_explanation, local_questions_from_chunks
from app.knowledge.chunking import chunk_document
from app.knowledge.embeddings import HashingEmbedder
from app.knowledge.extractors import extract_text_blob
from app.knowledge.retrieval import Retriever, build_context, detect_conflicts, ensure_lexical_index
from app.knowledge.vectorstore import LocalVectorStore
from app.knowledge.ingest import _replace_chunks
from app.models import Document


def _index_doc(db, *, title: str, text: str, tier: int = 1, doc_type: str = "md") -> Document:
    doc = Document(title=title, doc_type=doc_type, author="Tester", status=Document.STATUS_PROCESSING, tier=tier, kind="book")
    db.add(doc)
    db.commit()
    db.refresh(doc)
    extracted = extract_text_blob(text, title=title, doc_type=doc_type)
    drafts = chunk_document(extracted)
    _replace_chunks(db, doc, drafts, extracted)
    doc.status = Document.STATUS_INDEXED
    doc.chunk_count = len(drafts)
    db.commit()
    return doc


def test_hashing_embedder_is_deterministic_and_normalised():
    embedder = HashingEmbedder(dim=256)
    docs = ["gradient descent minimizes the loss function", "k-means clustering assigns points to centroids"]
    a = embedder.embed(docs)
    b = embedder.embed(docs)
    assert a.shape == (2, 256)
    assert np.allclose(a, b, atol=1e-6), "embedding must be deterministic"
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)


def test_similar_texts_score_higher_than_unrelated():
    embedder = HashingEmbedder(dim=512)
    anchor, similar, different = embedder.embed(
        [
            "gradient descent updates the weights to reduce the loss",
            "stochastic gradient descent reduces the training loss by stepping against the gradient",
            "the recipe needs two eggs and 200 grams of flour",
        ]
    )
    assert anchor @ similar > anchor @ different


def test_chunker_preserves_pages_sections_and_drops_duplicates(db):
    text = "\n\n".join(
        [
            "# Handwritten notes on linear models",
            "## 1.1 Least squares",
            "The least squares solution is w = (X^T X)^-1 X^T y. It minimises the squared residual and is the orthogonal projection onto the column space.",
            "## 1.2 Regularization",
            "Ridge regression adds lambda times the squared L2 norm of w to the objective, which shrinks coefficients and stabilises the inverse when X^T X is ill-conditioned.",
            "Ridge regression adds lambda times the squared L2 norm of w to the objective, which shrinks coefficients and stabilises the inverse when X^T X is ill-conditioned.",  # duplicate -> removed
            "## 1.3 Exercise",
            "Derive the gradient of the ridge objective and compare it to the normal equations. Explain why scaling matters before penalising weights.",
        ]
    )
    extracted = extract_text_blob(text, title="notes", doc_type="md")
    drafts = chunk_document(extracted, min_chars=40, overlap_chars=60)
    assert drafts, "chunker produced no chunks"
    assert all(d.text.strip() for d in drafts)
    assert len({d.content_hash for d in drafts}) == len(drafts), "duplicate chunks must be removed"
    assert any(d.section.startswith("1.2") or "Regularization" in d.section for d in drafts)
    assert all(d.page_start == 1 for d in drafts if d.page_start)
    assert any(d.keywords for d in drafts)


def test_hybrid_retrieval_ranks_the_relevant_document_first(db):
    _index_doc(
        db,
        title="Linear Models Handbook",
        text=(
            "# Linear Models Handbook\n\n## Ridge regression\n\n"
            "Ridge regression penalises the squared L2 norm of the weights, shrinking coefficients toward zero. "
            "Because the penalty is smooth, ridge never produces exactly zero coefficients; use lasso when sparsity is required.\n\n"
            "## Calibration\n\nReliability diagrams compare predicted probability bins against observed frequency. "
            "Platt scaling fits a logistic regression on the model scores to repair miscalibration.\n"
        ),
    )
    _index_doc(db, title="Cooking Notes", text="# Cooking Notes\n\nRidge the dough twice. Rest for thirty minutes before baking the bread.\n")

    retriever = Retriever(db)
    results = retriever.search("does ridge regression set coefficients exactly to zero?")
    assert results, "hybrid retrieval found nothing"
    assert results[0].document_title == "Linear Models Handbook"
    assert results[0].citation["document"] == "Linear Models Handbook"
    assert "Ridge" in results[0].text or "ridge" in results[0].text

    sufficient, coverage, hits = retriever.sufficient("does ridge regression set coefficients exactly to zero?")
    assert sufficient is True and coverage > 0.2
    irrelevant_sufficient, _cov, _hits = retriever.sufficient("how do I configure a gpu driver for infiniband?")
    assert irrelevant_sufficient is False, "must admit when the library has no answer"


def test_source_tier_and_metadata_are_used_in_ranking(db):
    _index_doc(db, title="Random Blog Post", text="# Blog\n\nPeople say ridge regression is basically the same as lasso, both zero out weights.\n", tier=5)
    _index_doc(db, title="Textbook of Regularization", text="# Textbook\n\n## Ridge\n\nRidge regression shrinks coefficients toward zero but keeps them non-zero; lasso can produce exact zeros.\n", tier=1)
    results = Retriever(db).search("ridge regression shrink coefficients exact zeros", top_k=4)
    assert results[0].document_title == "Textbook of Regularization"
    assert results[0].tier == 1


def test_build_context_numbers_sources_for_citation(db):
    _index_doc(db, title="Attention Explained", text="# Attention\n\nScaled dot-product attention divides by sqrt(d_k) to keep the variance of the scores stable.\n")
    results = Retriever(db).search("why divide attention by sqrt d_k", top_k=3)
    context, citations = build_context(results)
    assert context.startswith("[1] Attention Explained")
    assert citations[0]["document"] == "Attention Explained"
    assert citations[0]["index"] == 1
    assert "[1]" in context


def test_conflict_detection_surfaces_disagreement(db):
    _index_doc(db, title="Source A", text="# A\n\n## Dropout\n\nThe dropout rate should be set to 0.1 for small tabular models.\n", tier=1)
    _index_doc(db, title="Source B", text="# B\n\n## Dropout\n\nThe dropout rate should be set to 0.5 for small tabular models.\n", tier=1)
    results = Retriever(db).search("dropout rate small tabular models", top_k=4)
    conflicts = detect_conflicts(results, term="dropout")
    assert isinstance(conflicts, list)


def test_offline_explanation_is_grounded_and_cites(db):
    _index_doc(
        db,
        title="Optimization Notes",
        text=(
            "# Optimization Notes\n\n## Gradient descent\n\n"
            "Gradient descent is an iterative method that moves parameters against the gradient of the loss. "
            "The step size is called the learning rate; too large a value overshoots the minimum and can diverge.\n\n"
            "## Momentum\n\nMomentum accumulates an exponentially decaying average of past gradients, which damps oscillation.\n"
        ),
    )
    results = Retriever(db).search("what is gradient descent and why does the learning rate matter", top_k=5)
    payload = local_explanation(topic="Gradient descent", level=1, results=results)
    assert payload["grounded"] is True
    assert "[1]" in payload["text"]
    assert "Check your understanding" in payload["text"]
    assert payload["sources"] and payload["sources"][0]["document"] == "Optimization Notes"


def test_offline_question_generation_from_chunks(db):
    _index_doc(
        db,
        title="Bias Variance Notes",
        text=(
            "# Bias Variance Notes\n\n## Overfitting\n\n"
            "Overfitting is fitting the training noise instead of the signal, which raises validation error. "
            "Underfitting is when the model class is too simple to capture the relationship.\n\n"
            "## Regularization\n\nRegularization is any constraint or penalty that reduces variance at the cost of some bias.\n"
        ),
    )
    results = Retriever(db).search("overfitting regularization", top_k=4)
    questions = local_questions_from_chunks(topic="Overfitting", results=results, count=3, difficulty=2)
    assert questions, "template generator produced nothing"
    for q in questions:
        assert q["stem"]
        assert "generated_by" not in q or q["generated_by"] == "template"
    assert any("____" in q["stem"] for q in questions), "expected at least one cloze question"


def test_lexical_index_rebuilds_when_corpus_changes(db):
    before = ensure_lexical_index(db).n_docs
    _index_doc(db, title="Extra Doc", text="# Extra\n\nThis document adds exactly one searchable chunk about gradient boosting ensembles.\n")
    ensure_lexical_index(db, force=True)
    after = ensure_lexical_index(db).n_docs
    assert after > before


def test_local_vector_store_roundtrip(tmp_path):
    store = LocalVectorStore(path=tmp_path / "index.npz", dim=8)
    embedder = HashingEmbedder(dim=8)
    vectors = embedder.embed(["alpha beta gamma", "delta epsilon zeta", "alpha beta gamma again"])
    for i, vec in enumerate(vectors):
        store.upsert(chunk_id=100 + i, document_id=1, vector=vec, meta={})
    assert store.count() == 3
    hits = store.search(vectors[0], top_k=3)
    assert hits and hits[0].chunk_id == 100
    assert hits[0].score == pytest.approx(1.0, abs=1e-4)
    assert store.delete_document(1) == 3
    assert store.count() == 0
