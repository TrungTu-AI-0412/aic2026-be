"""Query rewriting: an LLM translates to English and strips prompt artifacts.

Queries arrive in Vietnamese wrapped in operator instructions - "hãy tìm trong
video đoạn có...", question forms, quotes, numbering. The vision side of
retrieval is English-centric (the SigLIP2 text tower, the BLIP ITM reranker), so
that wrapper is scored as if it described the scene. This turns each query into
a plain English visual description before it is encoded.

The endpoint is a network hop on the query path, which nothing else here is
allowed to be, so every failure mode - box down, timeout, malformed output -
returns None and the caller keeps its original text. A rewrite is an
improvement, never a dependency.

The speech stage deliberately does *not* use the rewrite: it matches Vietnamese
transcripts, dense and by term overlap, and English would drop the lexical half
to nothing. See `engine.retrieve`.
"""

import re
import time
from functools import lru_cache
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:  # pragma: no cover - import cycle: engine imports this module
    from app.retrieval.engine import RetrievalConfig, Timings

SYSTEM_PROMPT = (
    "You rewrite video-search queries for a CLIP-style image-text retrieval"
    " engine. For each numbered input query, output one line: the same number, a"
    " period, a space, then the rewritten query.\n"
    "Rules:\n"
    "- Translate to English. Keep proper nouns, brand names and numbers"
    " verbatim.\n"
    '- Describe only what is visible in the frame. Drop meta-instructions ("find'
    ' the moment", "in the video", "clip where"), question forms, quotes and'
    " numbering.\n"
    "- Do not add, guess or expand detail that is not in the input. Do not merge"
    " or split queries.\n"
    "- Output only the numbered lines, nothing else."
)

# One rewritten query per line, numbered as the input was. Tolerant about the
# separator, strict about the number: that is what the ordering check needs.
_LINE = re.compile(r"^\s*(\d+)\s*[.):]\s*(.+?)\s*$")


def rewrite_queries(
    texts: list[str], config: "RetrievalConfig", timings: "Timings"
) -> list[str] | None:
    """English, artifact-free forms of `texts`, in order - or None.

    None means nothing was rewritten: the step is off, unconfigured, or the call
    failed. The caller then keeps its original text and reports None, so an
    operator can tell a query the model left alone from one it never saw.

    Every query of a request is rewritten in one call. A TRAKE query is an
    overview plus N events, and N+1 round trips would put seconds on the clock
    for no gain.
    """
    if not config.rewrite_enabled or not config.rewrite_base_url or not texts:
        return None

    started = time.perf_counter()
    try:
        return list(
            _rewrite(
                tuple(texts),
                config.rewrite_base_url,
                config.rewrite_model,
                config.rewrite_api_key,
                config.rewrite_timeout_sec,
            )
        )
    except Exception:
        return None
    finally:
        timings.record("rewrite", started)


@lru_cache(maxsize=512)
def _rewrite(
    texts: tuple[str, ...],
    base_url: str,
    model: str,
    api_key: str,
    timeout: float,
) -> tuple[str, ...]:
    """Rewrite one batch, raising on any failure.

    Raising rather than falling back is what makes the cache correct: `lru_cache`
    does not memoise exceptions, so a successful rewrite is reused for the rest
    of the session and a failed one is retried the moment the box is back.
    """
    numbered = "\n".join(
        f"{index}. {text}" for index, text in enumerate(texts, start=1)
    )
    content = _post(
        base_url,
        {
            "model": model,
            "temperature": 0,
            "max_tokens": 64 * len(texts) + 64,
            # Qwen3 serves a thinking mode by default on some vLLM builds, and a
            # reasoning block ahead of the answer is tokens spent on latency.
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": numbered},
            ],
        },
        api_key,
        timeout,
    )
    return _parse(content, len(texts))


def _post(base_url: str, payload: dict, api_key: str, timeout: float) -> str:
    """The assistant message from an OpenAI-compatible chat completion."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    response = httpx.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json=payload,
        headers=headers,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def _parse(content: str, expected: int) -> tuple[str, ...]:
    """The numbered lines of `content`, in index order.

    All or nothing: a partial parse would silently shift a TRAKE event onto the
    wrong query, which ranks worse than not rewriting at all. Anything short of
    every index from 1 to `expected` raises and the caller falls back.
    """
    # Belt and braces. The competition endpoint emits no reasoning block with
    # thinking disabled, but a different server or model would.
    body = content.rsplit("</think>", 1)[-1]

    found: dict[int, str] = {}
    for line in body.splitlines():
        match = _LINE.match(line)
        if match:
            found[int(match.group(1))] = match.group(2).strip("\"'“”")

    if sorted(found) != list(range(1, expected + 1)) or not all(found.values()):
        raise ValueError(f"expected {expected} numbered queries, got {sorted(found)}")
    return tuple(found[index] for index in range(1, expected + 1))
