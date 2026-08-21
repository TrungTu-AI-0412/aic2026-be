"""Query rewriting: one LLM call, two forms of every query.

Queries arrive in Vietnamese wrapped in operator instructions - "hãy tìm trong
video đoạn có...", question forms, quotes, numbering. Retrieval then searches
two collections that want opposite things from that string:

- keyframe images, through the SigLIP2 text tower and the BLIP ITM reranker,
  which are English-centric and score the wrapper as if it described the scene;
- speech transcripts, which are Vietnamese and are matched dense *and* by term
  overlap, so translating for them would drop the lexical half to nothing -
  while leaving the wrapper in place makes `hãy`, `tìm`, `trong`, `video`,
  `đoạn`, `có` live BM25 terms scoring against transcripts.

So each query comes back as a `Rewrite`: translated for the image space, and
merely cleaned for the speech space. Both forms come out of the same call, which
is why this costs one round trip rather than two.

The endpoint is a network hop on the query path, which nothing else here is
allowed to be, so every failure mode - box down, timeout, malformed output -
returns None and the caller uses the query as typed for both. A rewrite is an
improvement, never a dependency.
"""

import re
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:  # pragma: no cover - import cycle: engine imports this module
    from app.retrieval.engine import RetrievalConfig, Timings


@dataclass(frozen=True)
class Rewrite:
    """One query in the two forms retrieval needs."""

    vision: str  # English: SigLIP2's text tower and the BLIP reranker
    speech: str  # Original language, artifacts stripped: Vietnamese transcripts


SYSTEM_PROMPT = (
    "You rewrite video-search queries for a retrieval engine that searches"
    " keyframe images and speech transcripts separately.\n"
    "For each numbered input query, output exactly one line: the same number, a"
    ' period, a space, the ENGLISH form, then " || ", then the CLEANED form.\n'
    "- ENGLISH form: the query translated to English, describing only what is"
    " visible in the frame. Keep proper nouns, brand names and numbers"
    " verbatim.\n"
    "- CLEANED form: the query in its ORIGINAL language with only the"
    " meta-instructions removed. Do not translate it, do not paraphrase it, do"
    " not reorder it, do not correct spelling. Keep every content word exactly"
    " as written.\n"
    '- From both forms drop meta-instructions ("find the moment", "in the'
    ' video", "the clip where", "hãy tìm", "đoạn có"), question framing, quotes'
    " and numbering.\n"
    "- Do not add, guess or expand detail that is not in the input. Do not merge"
    " or split queries.\n"
    "- Output only the numbered lines, nothing else."
)

# One line per query, numbered as the input was. Tolerant about the separator,
# strict about the number: that is what the ordering check needs.
_LINE = re.compile(r"^\s*(\d+)\s*[.):]\s*(.+?)\s*$")
# Divides the two forms on a line. Chosen because no natural query contains it.
_FORMS = " || "


def rewrite_queries(
    texts: list[str], config: "RetrievalConfig", timings: "Timings"
) -> list["Rewrite"] | None:
    """Both forms of each of `texts`, in order - or None.

    None means nothing was rewritten: the step is off, unconfigured, or the call
    failed. The caller then uses the query as typed for both forms and reports
    None, so an operator can tell a query the model left alone from one it never
    saw.

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
) -> tuple[Rewrite, ...]:
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
            # Two forms per query, measured at ~50 output tokens together.
            "max_tokens": 128 * len(texts) + 64,
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


def _parse(content: str, expected: int) -> tuple[Rewrite, ...]:
    """The numbered lines of `content`, in index order, split into both forms.

    All or nothing: a partial parse would silently shift a TRAKE event onto the
    wrong query, which ranks worse than not rewriting at all. Anything short of
    every index from 1 to `expected`, each carrying both forms, raises and the
    caller falls back.
    """
    # Belt and braces. The competition endpoint emits no reasoning block with
    # thinking disabled, but a different server or model would.
    body = content.rsplit("</think>", 1)[-1]

    found: dict[int, Rewrite] = {}
    for line in body.splitlines():
        match = _LINE.match(line)
        if not match:
            continue
        vision, separator, speech = match.group(2).partition(_FORMS)
        if not separator:
            continue
        found[int(match.group(1))] = Rewrite(_unquote(vision), _unquote(speech))

    if sorted(found) != list(range(1, expected + 1)) or not all(
        rewrite.vision and rewrite.speech for rewrite in found.values()
    ):
        raise ValueError(f"expected {expected} rewritten queries, got {sorted(found)}")
    return tuple(found[index] for index in range(1, expected + 1))


def _unquote(text: str) -> str:
    return text.strip().strip("\"'“”")
