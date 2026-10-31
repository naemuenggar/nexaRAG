"""Normalisation applied to every extracted page before it is stored.

Deliberately conservative: it repairs extraction artefacts but never drops words, so page text
stays faithful to the PDF and citation snippets can be matched back to it.
"""

import re
import unicodedata

# C0/C1 control characters except tab and newline. NUL in particular cannot be stored in
# PostgreSQL text columns.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
# "infor-\nmation" -> "information" (a lowercase letter must follow, so "COVID-\n19" is kept).
_HYPHENATED_BREAK = re.compile(r"(\w)-\n(?=[a-z])")
_SPACES = re.compile("[ \t\u00a0\u2000-\u200b\u202f\u205f\u3000]+")
_SPACE_AROUND_NEWLINE = re.compile(r" *\n *")
_MANY_NEWLINES = re.compile(r"\n{3,}")


def clean_text(raw: str) -> str:
    text = unicodedata.normalize("NFKC", raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub("", text)
    text = text.replace("�", "")
    text = _SPACES.sub(" ", text)
    text = _SPACE_AROUND_NEWLINE.sub("\n", text)
    text = _HYPHENATED_BREAK.sub(r"\1", text)
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip()
