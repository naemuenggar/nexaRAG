"""Test PDFs. Plain text PDFs come from the evaluation package; the rest are test-only."""

import io

from pypdf import PdfReader, PdfWriter

from rag_platform.evaluation.pdf import make_pdf

__all__ = ["make_encrypted_pdf", "make_malformed_pdf", "make_pdf"]


def make_encrypted_pdf(pages: list[str], user_password: str) -> bytes:
    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(make_pdf(pages))))
    writer.encrypt(user_password=user_password, owner_password="owner-secret", algorithm="AES-128")
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def make_malformed_pdf() -> bytes:
    """Has the PDF signature, so it passes upload validation, but is not a parseable PDF."""
    return b"%PDF-1.7\n1 0 obj\n<< /Type /Catalog /Pages 99 0 R\ngarbage without structure\n"
