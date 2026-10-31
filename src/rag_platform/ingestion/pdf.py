"""PDF validation and page-level text extraction (pypdf).

Parsing is CPU-bound and synchronous; callers run it in a worker thread.
"""

import io
import logging
from dataclasses import dataclass

from pypdf import PdfReader
from pypdf.errors import PdfReadError, PyPdfError

from rag_platform.ingestion.cleaning import clean_text

logger = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF-"
# The PDF spec allows junk before the header; readers accept it within the first 1 KiB.
MAGIC_SEARCH_WINDOW = 1024


@dataclass(frozen=True, slots=True)
class ParsedPage:
    number: int  # 1-based
    text: str


@dataclass(frozen=True, slots=True)
class ParsedPdf:
    page_count: int
    pages: list[ParsedPage]

    @property
    def total_chars(self) -> int:
        return sum(len(p.text) for p in self.pages)


class PdfParseError(Exception):
    """A permanent problem with the file itself; retrying will not help.

    ``message`` is safe to show to the document's owner.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def looks_like_pdf(head: bytes) -> bool:
    return PDF_MAGIC in head[:MAGIC_SEARCH_WINDOW]


def parse_pdf(data: bytes, *, max_pages: int) -> ParsedPdf:
    if not looks_like_pdf(data):
        raise PdfParseError("not_pdf", "The file is not a PDF.")

    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted and not reader.decrypt(""):
            # Owner-password-only PDFs decrypt with an empty user password; others cannot be read.
            raise PdfParseError("encrypted", "The PDF is password-protected.")
        page_count = len(reader.pages)
    except PdfParseError:
        raise
    except (PdfReadError, PyPdfError, ValueError, KeyError, TypeError, OSError) as exc:
        raise PdfParseError("malformed", "The PDF is damaged or not readable.") from exc
    except NotImplementedError as exc:  # e.g. unsupported encryption algorithm
        raise PdfParseError("unsupported", "The PDF uses an unsupported feature.") from exc

    if page_count == 0:
        raise PdfParseError("empty", "The PDF has no pages.")
    if page_count > max_pages:
        raise PdfParseError("too_many_pages", f"The PDF has more than {max_pages} pages.")

    pages: list[ParsedPage] = []
    failed_pages = 0
    for index in range(page_count):
        try:
            raw = reader.pages[index].extract_text() or ""
        except Exception:
            # One broken page shouldn't sink the whole document; it is kept as empty text.
            logger.warning("pdf_page_extraction_failed", extra={"page": index + 1}, exc_info=True)
            failed_pages += 1
            raw = ""
        pages.append(ParsedPage(number=index + 1, text=clean_text(raw)))

    if failed_pages == page_count:
        raise PdfParseError("malformed", "No page of the PDF could be read.")

    parsed = ParsedPdf(page_count=page_count, pages=pages)
    if parsed.total_chars == 0:
        raise PdfParseError(
            "no_text",
            "No extractable text found. Scanned PDFs need OCR, which is not supported yet.",
        )
    return parsed
