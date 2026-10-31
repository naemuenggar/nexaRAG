"""Heuristic detection of instructions embedded in retrieved document text.

This is a second line of defence. The first is structural: document text only ever appears
inside escaped ``<source>`` elements of the user message, and the system prompt says to treat it
as quoted data. Detection lets us warn the model about a specific source and log the attempt.
"""

import re

_PATTERNS = [
    r"\bignore\s+(?:all\s+|any\s+)?(?:the\s+)?(?:previous|prior|above|earlier)\s+"
    r"(?:instructions|prompts?|messages?|rules)",
    r"\bdisregard\s+(?:all\s+|any\s+)?(?:the\s+)?(?:previous|prior|above|earlier|system)\b",
    r"\bforget\s+(?:all\s+|everything\s+)?(?:you\s+were\s+told|previous\s+instructions)",
    r"\b(?:reveal|print|show|repeat|output)\s+(?:me\s+)?(?:the\s+|your\s+)?"
    r"(?:system\s+prompt|hidden\s+instructions|initial\s+instructions)",
    r"\byou\s+are\s+now\s+(?:a|an|in)\b",
    r"\bnew\s+instructions?\s*:",
    r"\b(?:system|assistant)\s*:\s",
    r"</?\s*(?:system|instructions?|source|sources)\s*>",
    r"\bdo\s+not\s+(?:cite|mention)\s+(?:the\s+)?sources?\b",
    r"\bact\s+as\s+(?:if\s+you\s+are\s+)?(?:a|an|the)\s+(?:different|unrestricted|jailbroken)",
]
_DETECTOR = re.compile("|".join(f"(?:{p})" for p in _PATTERNS), re.IGNORECASE)


def looks_like_injection(text: str) -> bool:
    return _DETECTOR.search(text) is not None
