"""Tokenisation for sparse retrieval: lowercase word tokens without common English stopwords.

No stemming: it would merge "accounts" and "accounting", and exact terms (product codes, names)
are what sparse retrieval is for. Dense retrieval covers paraphrases.
"""

import re
import unicodedata

_TOKEN = re.compile(r"\w+(?:[.'-]\w+)*")

STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been before
    being below between both but by can could did do does doing down during each few for
    from further had has have having he her here hers herself him himself his how i if in
    into is it its itself just me more most my myself no nor not of off on once only or
    other our ours ourselves out over own same she should so some such than that the their
    theirs them themselves then there these they this those through to too under until up
    very was we were what when where which while who whom why will with would you your yours
    yourself yourselves
    """.split()  # noqa: SIM905 - a word list reads better than a list literal
)


def tokenize(text: str) -> list[str]:
    normalised = unicodedata.normalize("NFKC", text).lower()
    return [t for t in _TOKEN.findall(normalised) if t not in STOPWORDS]
