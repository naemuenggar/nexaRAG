"""Builds small, valid text PDFs in memory (used for evaluation corpora and tests)."""

import io


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(pages: list[str]) -> bytes:
    """One page per string; each line of a string becomes a line of Helvetica text.

    An empty string produces a page with no text (like a scanned image page).
    """
    page_count = len(pages)
    font_id = 3
    first_page_id = 4
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        font_id: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    kids = []
    for index, text in enumerate(pages):
        page_id = first_page_id + 2 * index
        content_id = page_id + 1
        kids.append(f"{page_id} 0 R")
        lines = [f"({_escape(line)}) Tj T*" for line in text.splitlines()]
        stream = ("BT /F1 12 Tf 14 TL 72 720 Td " + " ".join(lines) + " ET").encode("latin-1")
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode()
        objects[content_id] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream"
        )
    objects[2] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {page_count} >>".encode()

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: dict[int, int] = {}
    for obj_id in sorted(objects):
        offsets[obj_id] = out.tell()
        out.write(f"{obj_id} 0 obj\n".encode() + objects[obj_id] + b"\nendobj\n")
    xref_at = out.tell()
    size = max(objects) + 1
    out.write(f"xref\n0 {size}\n0000000000 65535 f \n".encode())
    for obj_id in range(1, size):
        out.write(f"{offsets[obj_id]:010d} 00000 n \n".encode())
    out.write(f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode())
    return out.getvalue()
