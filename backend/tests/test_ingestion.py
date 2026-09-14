"""
Ingestion pipeline for real file formats: PDF (with page numbers), EPUB, HTML, Markdown, TXT,
plus the failure paths and the library HTTP API.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from app.config import settings
from app.db import session_scope
from app.knowledge.extractors import extract_epub, extract_file, extract_pdf, extract_text_blob
from app.knowledge.ingest import ingest_document
from app.knowledge.chunking import chunk_document
from app.models import Document, DocumentChunk


def write_pdf(path: Path, pages: list[str]) -> None:
    """Hand-built minimal but valid PDF (one text run per page) - no external writer dependency."""
    body: list[bytes] = []
    page_ids, content_specs = [], []
    next_id = 4
    for text in pages:
        content = f"BT /F1 11 Tf 72 720 Td ({text}) Tj ET".encode()
        content_specs.append((next_id, content))
        next_id += 1
        page_ids.append(next_id)
        next_id += 1
    body.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    body.append(f"2 0 obj\n<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>\nendobj\n".encode())
    body.append(b"3 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")
    for (cid, content), pid in zip(content_specs, page_ids):
        body.append(f"{cid} 0 obj\n<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream\nendobj\n")
        body.append(
            f"{pid} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {cid} 0 R "
            f"/Resources << /Font << /F1 3 0 R >> >> >>\nendobj\n".encode()
        )
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for chunk_bytes in body:
        offsets.append(len(out))
        out += chunk_bytes
    xref = len(out)
    count = len(body) + 1
    out += f"xref\n0 {count}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {count} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))


def write_epub(path: Path, chapters: list[tuple[str, str]]) -> None:
    opf = """<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
 <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
  <dc:title>Regularization Handbook</dc:title><dc:creator>A. Author</dc:creator>
 </metadata>
 <manifest><item id="c1" href="ch1.xhtml" media-type="application/xhtml+xml"/><item id="c2" href="ch2.xhtml" media-type="application/xhtml+xml"/></manifest>
 <spine><itemref idref="c1"/><itemref idref="c2"/></spine>
</package>"""
    container = """<?xml version="1.0"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">
 <rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>
</container>"""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr("META-INF/container.xml", container)
        zf.writestr("OEBPS/content.opf", opf)
        for name, (title, text) in zip(["OEBPS/ch1.xhtml", "OEBPS/ch2.xhtml"], chapters):
            zf.writestr(
                name,
                f"<html><head><title>{title}</title></head><body><h1>{title}</h1><p>{text}</p></body></html>",
            )


def test_pdf_extraction_keeps_page_numbers(tmp_path):
    pdf = tmp_path / "book.pdf"
    write_pdf(pdf, ["Chapter 1. Gradient descent minimizes the loss function by taking negative gradient steps."] * 1 + ["The learning rate controls the step size and can cause divergence."] * 3)
    doc = extract_pdf(pdf)
    assert doc.page_count == 4
    assert all(page.page_number == index + 1 for index, page in enumerate(doc.pages))
    assert "Gradient descent" in doc.pages[0].text


def test_epub_extraction_uses_spine_and_titles(tmp_path):
    epub = tmp_path / "book.epub"
    write_epub(
        epub,
        [
            ("Ridge and lasso", "Ridge regression shrinks coefficients toward zero but rarely reaches it exactly, while lasso can produce exact zeros because of the non-differentiable L1 penalty."),
            ("Pipelines", "A scikit-learn pipeline fits every transformer inside the cross-validation loop, which is the structural defence against leakage."),
        ],
    )
    doc = extract_epub(epub)
    assert doc.title == "Regularization Handbook"
    assert doc.author == "A. Author"
    assert len(doc.pages) == 2
    assert "lasso" in doc.pages[0].text.lower()
    assert "leakage" in doc.pages[1].text.lower()


def test_html_and_markdown_and_dispatch(tmp_path):
    html = tmp_path / "page.html"
    html.write_text(
        "<html><head><title>Attention notes</title></head><body><nav>menu junk</nav>"
        "<article><h2>Scaled dot product</h2><p>Dividing by sqrt(d_k) keeps the variance of the attention scores stable so the softmax does not saturate.</p></article>"
        "<footer>cookie banner</footer></body></html>",
        encoding="utf-8",
    )
    doc = extract_file(html)
    assert doc.title == "Attention notes"
    assert "sqrt(d_k)" in doc.pages[0].text
    assert "menu junk" not in doc.pages[0].text

    md = tmp_path / "notes.md"
    md.write_text("# My notes\n\nauthor: Jane Doe\n\n## Section one\n\nBatch normalisation standardises activations per feature using batch statistics.\n", encoding="utf-8")
    md_doc = extract_file(md)
    assert md_doc.author == "Jane Doe"

    txt = tmp_path / "raw.txt"
    txt.write_text("Dropout randomly zeroes activations during training only.", encoding="utf-8")
    assert "Dropout" in extract_file(txt).pages[0].text

    bin_file = tmp_path / "mystery.bin"
    bin_file.write_bytes(b"\x00\x01\x02")
    with pytest.raises(ValueError):
        extract_file(bin_file)


def test_text_blob_marks_scan_heavy_short_doc(db, tmp_path):
    """A file with no extractable text must end up in Error status with a readable message."""
    empty = tmp_path / "empty.md"
    empty.write_text("# Title\n", encoding="utf-8")
    doc = Document(title="empty", doc_type="md", file_path=str(empty), status=Document.STATUS_UPLOADING)
    db.add(doc)
    db.commit()
    result = ingest_document(db, doc.id)
    db.commit()
    db.rollback()
    assert result.status == "error"
    assert "text" in result.error.lower() or "chunk" in result.error.lower()
    from sqlalchemy import select

    assert db.execute(select(Document.status).where(Document.id == doc.id)).scalar_one() == Document.STATUS_ERROR


def test_ingest_marks_topics_and_counts_chunks(db, fast_seeded):
    text = "\n\n".join(
        [
            "# Deep learning notes",
            "## Backpropagation",
            "Backpropagation applies the chain rule backwards through the computation graph, reusing activation gradients so each parameter receives its partial derivative.",
            "## Gradient clipping",
            "Gradient clipping rescales the update when the gradient norm exceeds a threshold, which prevents exploding gradients in recurrent networks.",
            "## Feature engineering",
            "Feature hashing maps raw strings into a fixed-width vector space, trading collisions for memory.",
        ]
    )
    blob = extract_text_blob(text, title="dl notes", doc_type="md")
    assert len(blob.pages) >= 1
    drafts = chunk_document(blob)
    assert drafts
    assert any(d.section for d in drafts), "sections must be captured from markdown headings"
    assert all(d.text for d in drafts)


def test_library_api_upload_index_search_delete(auth_client, db, tmp_path):
    client = auth_client
    pdf = tmp_path / "uploaded.pdf"
    write_pdf(
        pdf,
        [
            "Chapter 1 Intro. Early stopping halts training when validation loss stops improving, which constrains how far parameters travel.",
            "Chapter 1 Intro. Weight decay penalises the squared norm of the weights and is equivalent to a Gaussian prior over parameters.",
            "Chapter 1 Intro. Momentum accumulates an exponentially decaying average of past gradients to damp oscillations in the update.",
        ],
    )
    content = pdf.read_bytes()
    response = client.post(
        "/api/library/upload",
        files={"file": ("handbook.pdf", io.BytesIO(content), "application/pdf")},
        data={"title": "Uploaded Handbook", "author": "Test Author", "kind": "book", "tier": "1", "wait": "true"},
    )
    assert response.status_code == 201, response.text
    payload = response.json()
    document_id = payload["document"]["id"]
    assert payload["ingest"]["status"] == "indexed", payload["ingest"]
    assert payload["document"]["page_count"] == 3
    assert payload["document"]["chunk_count"] >= 1

    listing = client.get("/api/library/documents").json()
    match = [d for d in listing["items"] if d["id"] == document_id]
    assert match and match[0]["status"] == "indexed" and match[0]["author"] == "Test Author"

    search = client.get("/api/library/search", params={"q": "weight decay gaussian prior"}).json()
    assert search["count"] >= 1
    assert any(hit["document_id"] == document_id for hit in search["results"])

    detail = client.get(f"/api/library/documents/{document_id}").json()
    assert detail["chunks"], "chunk inspector must show chunks"
    first_page = detail["chunks"][0]["page_start"]
    assert isinstance(first_page, int) and first_page >= 1

    patched = client.patch(f"/api/library/documents/{document_id}", json={"notes": "read ch1", "tags": ["ml"]}).json()
    assert patched["notes"] == "read ch1"

    reindexed = client.post(f"/api/library/documents/{document_id}/index", params={"wait": True}).json()
    assert reindexed["ingest"]["status"] == "indexed"

    assert client.delete(f"/api/library/documents/{document_id}").json()["deleted"] == document_id
    with session_scope() as session:
        assert session.get(Document, document_id) is None
        assert session.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).count() == 0


def test_unsupported_upload_is_rejected(auth_client):
    response = auth_client.post(
        "/api/library/upload",
        files={"file": ("evil.xyz", io.BytesIO(b"not a document"), "application/octet-stream")},
        data={"title": "nope"},
    )
    assert response.status_code == 415
    assert "Unsupported file type" in response.json()["detail"]


def test_document_status_flow_is_visible(auth_client, db, tmp_path):
    client = auth_client
    """Uploading -> Processing -> Indexed/Error must be real state transitions, not decoration."""
    from app.models import Document

    doc = Document(title="ghost", doc_type="md", status=Document.STATUS_UPLOADING, file_path="/nonexistent.md")
    db.add(doc)
    db.commit()
    listing = client.get("/api/library/documents").json()
    entry = next(d for d in listing["items"] if d["id"] == doc.id)
    assert entry["status"] == "uploading"

    result = client.post(f"/api/library/documents/{doc.id}/index", params={"wait": True}).json()
    assert result["ingest"]["status"] == "error"
    assert "no file" in (result["document"]["error"] or "").lower()
    db.rollback()
    from sqlalchemy import select as _select
    assert _select  # noqa
    assert db.execute(_select(Document.status).where(Document.id == doc.id)).scalar_one() == Document.STATUS_ERROR

    good = tmp_path / "good.md"
    good.write_text(
        "# Learning rate warmup\n\n## Why\n\nWarmup starts with a small learning rate while second-order structure is unknown, "
        "which prevents early divergence in transformer training.\n\n## Schedule\n\nA linear warmup over a few percent of steps "
        "followed by cosine decay is the boring reliable default.\n",
        encoding="utf-8",
    )
    doc2 = Document(title="warmup notes", doc_type="md", status=Document.STATUS_UPLOADING, file_path=str(good), owner_id=None)
    db.add(doc2)
    db.commit()
    second = client.post(f"/api/library/documents/{doc2.id}/index", params={"wait": True}).json()
    assert second["ingest"]["status"] == "indexed", second
    assert second["document"]["chunk_count"] >= 1
    assert second["document"]["word_count"] > 20

    # and the queued path reports progress instead of pretending to be done
    third = client.post(
        "/api/library/upload",
        files={"file": ("queued.md", io.BytesIO(b"# Queued\n\n" + b"Exponential moving averages smooth the gradient estimates in Adam. " * 12), "text/markdown")},
        data={"title": "Queued doc", "kind": "notes"},
    ).json()
    assert third["ingest"]["queued"] is True
    assert third["document"]["status"] in {"uploading", "processing", "indexed"}


def test_library_search_spans_all_documents(auth_client, db, tmp_path):
    for name, body in (
        ("beam.md", "# Beam search\n\nBeam search keeps the top B partial sequences at every step, trading compute for a better approximation of argmax over sequences.\n"),
        ("quant.md", "# Quantisation\n\nInt8 quantisation stores weights in 8-bit integers, cutting memory traffic which dominates decode latency.\n"),
    ):
        path = tmp_path / name
        path.write_text(body, encoding="utf-8")
        from app.models import Document

        doc = Document(title=name, doc_type="md", status=Document.STATUS_UPLOADING, file_path=str(path))
        db.add(doc)
        db.commit()
        auth_client.post(f"/api/library/documents/{doc.id}/index", params={"wait": True})
    hits = auth_client.get("/api/library/search", params={"q": "beam search top B partial sequences"}).json()["results"]
    assert hits
    assert "beam" in hits[0]["snippet"].lower() or hits[0]["match_type"] == "document"
    stats = auth_client.get("/api/library/stats").json()
    assert stats["chunks"] >= 2
    assert stats["lexical_docs"] >= 2
    assert stats["vector_store"]["vectors"] >= 2


def test_web_url_validation_without_network(auth_client):
    """URL ingestion validates the target; it must not fetch arbitrary hosts in tests."""
    bad = auth_client.post("/api/library/url", json={"url": "ftp://example.com/x"}).json()
    assert "http" in str(bad).lower() or "detail" in bad
    private = auth_client.post("/api/library/url", json={"url": "http://127.0.0.1:9999/x"}).json()
    assert "detail" in private
