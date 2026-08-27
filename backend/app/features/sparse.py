"""Sparse (lexical) vectors for hybrid retrieval.

Dense embeddings are the wrong tool for the part of a query that names things.
SigLIP2 has no useful representation of "Nguyễn Xuân Son", "31/12/2025" or
"Hezbollah", yet those are exactly the tokens a competition query hangs on, and
exactly what is written across the lower third of a news broadcast. A sparse
vector indexed alongside the dense one gives those tokens somewhere to match.

Term frequencies are emitted raw. Qdrant applies the IDF part server-side when
the sparse vector is configured with `Modifier.IDF`, which makes the scoring
BM25-equivalent without this process needing corpus statistics it cannot have
while streaming a manifest.

Nothing here is Qdrant-specific: the output is a plain `SparseVector` of
indices and values, and `app/vector_store/` decides what to do with it.
"""

import math
import re
import unicodedata
import zlib
from dataclasses import dataclass, field

# Vietnamese is written as space-separated syllables, and this pipeline has no
# word segmenter, so a syllable is the unit. Matching "sông Cửu Long" then
# relies on all three syllables being present, which is the behaviour a
# segmenter would approximate anyway.
_TOKEN = re.compile(r"[0-9\w]+", re.UNICODE)

# Long strings of digits are timestamps and phone numbers burned into the
# broadcast overlay; they match nothing a query would ask for and only inflate
# the vector.
_MAX_TOKEN_LEN = 24


@dataclass(frozen=True)
class SparseVector:
    indices: list[int] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.indices)


def strip_diacritics(text: str) -> str:
    """Fold Vietnamese diacritics, including the đ/d distinction.

    This exists because of a measured asymmetry between the two text sources.
    OCR on this corpus fails almost exclusively on diacritics — `đường` is
    read as `dường`, `đối` as `dối` — and folding merges the two spellings
    back together. ASR fails on consonants instead (`sục` for `sụt`, `lúng`
    for `lún`), which folding cannot repair.

    So the folded variant recovers most OCR damage and none of the ASR
    damage. It is emitted alongside the exact token rather than replacing it,
    because folding also merges genuinely different words.
    """
    decomposed = unicodedata.normalize("NFD", text)
    without_marks = "".join(
        char for char in decomposed if unicodedata.category(char) != "Mn"
    )
    return without_marks.replace("đ", "d").replace("Đ", "D")


def tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFC", text).lower()
    return [
        token
        for token in _TOKEN.findall(normalized)
        if len(token) <= _MAX_TOKEN_LEN
    ]


def token_index(token: str) -> int:
    """Map a token to a stable uint32 slot.

    CRC32 rather than `hash()`: Python salts string hashing per process, so
    `hash()` would give a collection ingested today a different vocabulary
    from the query encoded tomorrow, and every lexical match would silently
    disappear.
    """
    return zlib.crc32(token.encode("utf-8")) & 0x7FFFFFFF


def encode(*texts: str, fold_diacritics: bool = True) -> SparseVector:
    """Build one sparse vector from any number of text fields.

    Fields are pooled rather than concatenated with separators because term
    frequency is all that survives into the vector anyway.

    Raw counts, which is right for a *query*: scaling every value in a query
    vector by the same factor cannot reorder anything. Points are a different
    matter — see `encode_document`.
    """
    counts = _counts(texts, fold_diacritics)
    return _to_vector(counts)


def encode_document(*texts: str, fold_diacritics: bool = True) -> SparseVector:
    """The same vector, scaled to unit length, for a point being ingested.

    WHY THIS EXISTS

    Qdrant's `Modifier.IDF` supplies the IDF half of BM25 server-side. The
    term-frequency half arrives from here, and as raw counts it has no length
    normalisation at all, so a point's score grows with how much text it
    carries. On this corpus that is not a subtle effect: a lecture slide packed
    with words outscored the three-word ticker that actually answered the
    query, at the highest score in the run.

    Measured over 200 cross-source queries (index built from EasyOCR's
    reading, queried with the VLM's reading of the same frame, so a hit is
    independent evidence rather than the tokeniser marking its own work):

        weighting        content@1   content@10   MRR    top-1 text length
        raw counts         0.625        0.720    0.651    52 tokens median
        L2-normalised      0.735        0.795    0.753    25
        BM25 k1=1.2 b=.75  0.740        0.815    0.762    27

    L2 is taken over BM25 despite the marginally lower numbers. BM25's
    saturation term needs the corpus average document length, which this
    process cannot know while streaming a manifest, and a constant baked in
    here would silently rot the moment the corpus or the field mix changed —
    a failure that produces slightly worse ranking rather than an error.
    Unit length needs nothing but the vector itself, and recovers 96% of the
    gain.

    The distinction is document-side only. `encode` stays raw for queries.
    """
    counts = _counts(texts, fold_diacritics)
    norm = math.sqrt(sum(value * value for value in counts.values()))
    if norm:
        counts = {index: value / norm for index, value in counts.items()}
    return _to_vector(counts)


def _counts(texts: tuple[str, ...], fold_diacritics: bool) -> dict[int, float]:
    counts: dict[int, float] = {}

    for text in texts:
        if not text:
            continue
        for token in tokenize(text):
            counts[token_index(token)] = counts.get(token_index(token), 0.0) + 1.0
            if fold_diacritics:
                folded = strip_diacritics(token)
                if folded != token:
                    slot = token_index(folded)
                    # Half weight: the folded form is a recall aid, and giving
                    # it the same weight as the exact token would let a
                    # diacritic-blind match outrank an exact one.
                    counts[slot] = counts.get(slot, 0.0) + 0.5

    return counts


def _to_vector(counts: dict[int, float]) -> SparseVector:
    if not counts:
        return SparseVector()

    ordered = sorted(counts.items())
    return SparseVector(
        indices=[index for index, _ in ordered],
        values=[value for _, value in ordered],
    )
