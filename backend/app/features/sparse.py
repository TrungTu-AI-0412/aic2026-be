"""Sparse (lexical) vectors for hybrid retrieval.

Dense embeddings are the wrong tool for the part of a query that names things.
SigLIP2 has no useful representation of "Nguyễn Xuân Son", "31/12/2025" or
"Hezbollah", yet those are exactly the tokens a competition query hangs on, and
exactly what is written across the lower third of a news broadcast. A sparse
vector indexed alongside the dense one gives those tokens somewhere to match.

Supports two sparse vector representation methods:
1. "bm25": Token frequencies with CRC32 hashing and Vietnamese diacritic folding.
   Term frequencies are emitted raw. Qdrant applies the IDF part server-side when
   the sparse vector is configured with `Modifier.IDF`, which makes the scoring
   BM25-equivalent without this process needing corpus statistics.
2. "splade": Neural Sparse Lexical and Expansion Model (e.g. naver/splade-cocondenser-ensembledistil)
   where activations are computed via log(1 + ReLU(MLM_logits)) over subword vocabulary tokens.
   Because SPLADE weights already factor in term importance and semantic expansion,
   Qdrant queries use standard dot product without Modifier.IDF.

Nothing here is Qdrant-specific: the output is a plain `SparseVector` of
indices and values, and `app/vector_store/` decides what to do with it.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
import re
from typing import Any, Literal
import unicodedata
import zlib

# Vietnamese is written as space-separated syllables, and this pipeline has no
# word segmenter, so a syllable is the unit. Matching "sông Cửu Long" then
# relies on all three syllables being present, which is the behaviour a
# segmenter would approximate anyway.
_TOKEN = re.compile(r"[0-9\w]+", re.UNICODE)

# Long strings of digits are timestamps and phone numbers burned into the
# broadcast overlay; they match nothing a query would ask for and only inflate
# the vector.
_MAX_TOKEN_LEN = 24

DEFAULT_SPLADE_MODEL = "naver/splade-cocondenser-ensembledistil"
SparseMethod = Literal["bm25", "splade"]


@dataclass(frozen=True)
class SparseVector:
    indices: list[int] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.indices)


# =============================================================================
# BM25 Lexical Hashing Implementation
# =============================================================================


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


def encode_bm25(*texts: str, fold_diacritics: bool = True) -> SparseVector:
    """Build one BM25 sparse vector from any number of text fields.

    Fields are pooled rather than concatenated with separators because term
    frequency is all that survives into the vector anyway.
    """
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

    if not counts:
        return SparseVector()

    ordered = sorted(counts.items())
    return SparseVector(
        indices=[index for index, _ in ordered],
        values=[value for _, value in ordered],
    )


# =============================================================================
# SPLADE Neural Sparse Implementation
# =============================================================================


@dataclass(frozen=True)
class _SpladeRuntime:
    tokenizer: Any
    model: Any
    torch: Any
    device: Any


@lru_cache(maxsize=None)
def _load_splade_runtime(model_id: str = DEFAULT_SPLADE_MODEL) -> _SpladeRuntime:
    try:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "SPLADE dependencies are missing; install torch and transformers"
        ) from exc

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForMaskedLM.from_pretrained(model_id)
        model = model.to(device).eval()
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"cannot load SPLADE model '{model_id}': {exc}"
        ) from exc

    return _SpladeRuntime(tokenizer, model, torch, device)


def encode_splade(
    *texts: str,
    model_id: str = DEFAULT_SPLADE_MODEL,
    threshold: float = 1e-4,
    max_length: int = 512,
    top_k: int | None = None,
) -> SparseVector:
    """Build one SPLADE sparse vector from one or more text inputs.

    Computes activation weights w_j = max_t log(1 + relu(MLM(h_t)_j)) for each
    vocabulary token. Multiple input texts are pooled by max activation across
    texts.
    """
    valid_texts = [text.strip() for text in texts if text and text.strip()]
    if not valid_texts:
        return SparseVector()

    runtime = _load_splade_runtime(model_id)
    inputs = runtime.tokenizer(
        valid_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    inputs = {k: v.to(runtime.device) for k, v in inputs.items()}

    with runtime.torch.inference_mode():
        outputs = runtime.model(**inputs)
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        # Standard SPLADE activation: log(1 + relu(logits))
        relu_log = runtime.torch.log1p(runtime.torch.relu(logits))
        # Mask padding tokens
        attention_mask = inputs["attention_mask"].unsqueeze(-1)  # [batch_size, seq_len, 1]
        relu_log = relu_log * attention_mask
        # Max-pool over tokens in each sequence
        doc_reps, _ = runtime.torch.max(relu_log, dim=1)  # [batch_size, vocab_size]
        # Max-pool across multiple texts
        if doc_reps.shape[0] > 1:
            doc_rep, _ = runtime.torch.max(doc_reps, dim=0)
        else:
            doc_rep = doc_reps[0]

        # Extract non-zero activations exceeding threshold
        non_zero_mask = doc_rep > threshold
        indices = runtime.torch.nonzero(non_zero_mask, as_tuple=True)[0]
        values = doc_rep[indices]

        if top_k is not None and top_k > 0 and len(indices) > top_k:
            topk_values, topk_sub_idx = runtime.torch.topk(values, k=top_k)
            indices = indices[topk_sub_idx]
            values = topk_values

        indices_list = indices.cpu().tolist()
        values_list = [round(float(v), 4) for v in values.cpu().tolist()]

    if not indices_list:
        return SparseVector()

    ordered = sorted(zip(indices_list, values_list))
    return SparseVector(
        indices=[int(idx) for idx, _ in ordered],
        values=[float(val) for _, val in ordered],
    )


# =============================================================================
# Unified Entrypoint
# =============================================================================


def encode(
    *texts: str,
    method: str = "bm25",
    fold_diacritics: bool = True,
    model_id: str = DEFAULT_SPLADE_MODEL,
    threshold: float = 1e-4,
    max_length: int = 512,
    top_k: int | None = None,
) -> SparseVector:
    """Build one sparse vector from any number of text fields using BM25 or SPLADE.

    Args:
        *texts: Text strings to encode.
        method: "bm25" (lexical hashing + TF) or "splade" (neural masked LM expansion).
        fold_diacritics: (BM25 only) Whether to fold Vietnamese diacritics with 0.5 weight.
        model_id: (SPLADE only) Model name or local HF cache path.
        threshold: (SPLADE only) Minimum weight to retain in the sparse vector.
        max_length: (SPLADE only) Maximum tokenizer sequence length.
        top_k: (SPLADE only) Maximum number of non-zero activations to retain.
    """
    if method == "bm25":
        return encode_bm25(*texts, fold_diacritics=fold_diacritics)
    elif method == "splade":
        return encode_splade(
            *texts,
            model_id=model_id,
            threshold=threshold,
            max_length=max_length,
            top_k=top_k,
        )
    else:
        raise ValueError(
            f"unknown sparse encoding method '{method}'; expected 'bm25' or 'splade'"
        )

