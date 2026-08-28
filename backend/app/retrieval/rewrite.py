"""Query rewriting: two concurrent LLM calls, two forms of every query.

Queries arrive in Vietnamese wrapped in narration about the video or the search
itself - "đoạn video mô tả...", "tìm phân cảnh với...", question forms, quotes.
Retrieval then searches two collections that want opposite things from that
string, so each query is prepared twice:

- keyframe images, through the SigLIP2 text tower and the BLIP ITM reranker.
  Both are English-centric and score the narration as if it described the scene,
  and the text tower reads exactly **64 tokens** (`padding="max_length"` in
  `features.multimodal.embed_text`) - about 45 English words. A literal
  translation of a 700-character KIS description runs well past that and is cut
  without warning, losing the tail where the distinguishing detail sits. So this
  side gets a short English *caption*.
- speech transcripts, which are Vietnamese and are matched dense *and* by term
  overlap, so translating for them would drop the lexical half to nothing -
  while leaving the narration in place makes `đoạn video`, `phân cảnh`, `tìm`
  live BM25 terms scoring against transcripts. So this side keeps the original
  wording with the narration *deleted*, never reworded.

The two jobs get **separate prompts on separate concurrent calls**. Asking for
both on one line was measurably worse at both: the caption rules bled into the
deletion rules and the model variously deleted the subject of the query
("lễ hội đèn lồng", "4 phi hành gia mặc áo đen") or deleted nothing at all.
Split, each call does its own job cleanly, and the wall clock is the slower of
the two rather than their sum.

The endpoint is a network hop on the query path, which nothing else here is
allowed to be, so every failure mode - box down, timeout, malformed output -
falls back to the query as typed. The two halves fail independently: a caption
that does not arrive does not cost the cleaned form, which is the whole point of
splitting them. A rewrite is an improvement, never a dependency.

`decompose.py` reuses `CLEAN_PROMPT` and `_call` from here for the query it
splits into events, but not `CAPTION_PROMPT`: a decomposed event is searched
alone against single frames, and captioning it alone loses the scene it belongs
to. See that module for which of its calls run beside which.
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:  # pragma: no cover - import cycle: engine imports this module
    from app.retrieval.engine import RetrievalConfig, Timings


@dataclass(frozen=True)
class Rewrite:
    """One query in the two forms retrieval needs."""

    vision: str  # English caption: SigLIP2's text tower and the BLIP reranker
    speech: str  # Original language, narration deleted: Vietnamese transcripts


CAPTION_PROMPT = (
    "You turn a video-search query into a caption for an image search over"
    " single frames. The query is an operator's description of a moment they"
    " want to find, wrapped in narration about the video or the search itself.\n"
    "English, AT MOST 40 words - count them, a longer caption is cut off and"
    " wasted. Spend those words on what a camera records: the camera angle or"
    " shot type when the query states one (overhead, top-down, head-on,"
    " close-up, wide), how many of each thing, colours, clothing, objects, where"
    " things sit relative to each other, the setting, expressions, the action.\n"
    "Never drop a visible detail just to be shorter, and never invent one - if"
    " the query says nothing about the camera, do not mention the camera. Leave"
    " out only what no frame can show: the narration about the video and the"
    " search itself. Never merge or split queries.\n"
    "Output one line per input: the same number, a period, a space, then the"
    " caption. Nothing else."
)

CLEAN_PROMPT = (
    "You strip search-narration out of a video-search query. The result is fed"
    " to a keyword and semantic search over speech transcripts, so it must stay"
    " in the language it arrived in.\n"
    "This is a DELETION task. Copy each numbered input back word for word,"
    " removing only the words that refer to the video, the clip, the scene, the"
    " frame or the act of searching. Never translate, paraphrase, reorder,"
    " summarise, shorten or respell anything you keep. Never delete a word that"
    " names something in the world - a person, place, object, colour, count,"
    " action or camera angle. You are striking a few words off the front of a"
    " sentence, never the sentence itself. When in doubt, keep it.\n"
    "Examples of the only kind of edit allowed:\n"
    '  "Đoạn video về tường thuật một cuộc đua xe đạp." -> "tường thuật một'
    ' cuộc đua xe đạp."\n'
    '  "Đoạn clip bắt đầu với hình ảnh 4 phi hành gia mặc áo đen." -> "4 phi'
    ' hành gia mặc áo đen."\n'
    '  "Tìm phân cảnh với góc quay từ trên cao xuống dõi theo các tay đua." ->'
    ' "góc quay từ trên cao xuống dõi theo các tay đua."\n'
    '  "Đoạn clip ghi lại một lễ hội đèn lồng." -> "một lễ hội đèn lồng."\n'
    '  "Trong khung hình gồm có 3 tay đua đạp thành một đường thẳng." -> "3 tay'
    ' đua đạp thành một đường thẳng."\n'
    '  "Cả hai đều mỉm cười, tỏ vẻ thích thú." -> unchanged, nothing to delete.\n'
    "Output one line per input: the same number, a period, a space, then the"
    " stripped query. Nothing else."
)

# One line per query, numbered as the input was. Tolerant about the separator,
# strict about the number: that is what the ordering check needs.
_LINE = re.compile(r"^\s*(\d+)\s*[.):]\s*(.+?)\s*$")


def rewrite_queries(
    texts: list[str], config: "RetrievalConfig", timings: "Timings"
) -> list["Rewrite"] | None:
    """Both forms of each of `texts`, in order - or None if neither arrived.

    Every query of the request goes in one batch per call. A TRAKE query is an
    overview plus N events, and per-query round trips would put seconds on the
    clock for no gain.

    None means nothing was rewritten: the step is off, unconfigured, or both
    calls failed. The caller then uses the query as typed and reports None, so
    an operator can tell a query the model left alone from one it never saw.
    """
    if not config.rewrite_enabled or not config.rewrite_base_url or not texts:
        return None

    batch = tuple(texts)
    started = time.perf_counter()
    try:
        # Both calls at once: the caption is short and the cleaned form is nearly
        # as long as the query, so run sequentially this would cost the sum of
        # the two rather than the slower one.
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = [
                pool.submit(_attempt, prompt, batch, config, budget(batch))
                for prompt, budget in (
                    (CAPTION_PROMPT, _caption_budget),
                    (CLEAN_PROMPT, _clean_budget),
                )
            ]
            captions, cleaned = (task.result() for task in pending)

        if captions is None and cleaned is None:
            return None
        # Each half falls back to the query as typed on its own, which is what
        # separate calls buy: a missing caption must not cost the cleaned form.
        return [
            Rewrite(vision=caption, speech=speech)
            for caption, speech in zip(captions or batch, cleaned or batch)
        ]
    finally:
        timings.record("rewrite", started)


def _caption_budget(texts: tuple[str, ...]) -> int:
    """Room for a 40-word caption per query, whatever the query's length."""
    return 64 * len(texts) + 64


def _clean_budget(texts: tuple[str, ...]) -> int:
    """Room to echo every query back, minus a few words.

    Sized from the input, not the query count: the cleaned form is the query
    itself less its narration, so a 700-character KIS description needs several
    hundred tokens on its own. Vietnamese runs about two characters per token in
    this tokenizer.
    """
    return sum(len(text) for text in texts) // 2 + 32 * len(texts) + 64


def _attempt(
    system: str, texts: tuple[str, ...], config: "RetrievalConfig", max_tokens: int
) -> tuple[str, ...] | None:
    """One call's results, or None if it failed for any reason at all."""
    try:
        return _call(
            system,
            texts,
            max_tokens,
            len(texts),
            config.rewrite_base_url,
            config.rewrite_model,
            config.rewrite_api_key,
            config.rewrite_timeout_sec,
        )
    except Exception:
        return None


@lru_cache(maxsize=512)
def _call(
    system: str,
    texts: tuple[str, ...],
    max_tokens: int,
    expected: int | None,
    base_url: str,
    model: str,
    api_key: str,
    timeout: float,
) -> tuple[str, ...]:
    """Send one batch under one prompt, raising on any failure.

    Raising rather than falling back is what makes the cache correct: `lru_cache`
    does not memoise exceptions, so a successful call is reused for the rest of
    the session and a failed one is retried the moment the box is back.
    """
    numbered = "\n".join(
        f"{index}. {text}" for index, text in enumerate(texts, start=1)
    )
    content, finish_reason = _post(
        base_url,
        {
            "model": model,
            "temperature": 0,
            "max_tokens": max_tokens,
            # Qwen3 serves a thinking mode by default on some vLLM builds, and a
            # reasoning block ahead of the answer is tokens spent on latency.
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": numbered},
            ],
        },
        api_key,
        timeout,
    )
    # A truncated answer still parses - the early lines are intact - so without
    # this the last query comes back as half a sentence and is searched as if the
    # operator had typed it that way.
    if finish_reason != "stop":
        raise ValueError(f"rewrite stopped on {finish_reason!r}, not a stop token")
    return _parse(content, expected)


def _post(
    base_url: str, payload: dict, api_key: str, timeout: float
) -> tuple[str, str]:
    """The assistant message and its finish reason, from a chat completion."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    response = httpx.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json=payload,
        headers=headers,
        timeout=timeout,
    )
    response.raise_for_status()
    choice = response.json()["choices"][0]
    return choice["message"]["content"], choice["finish_reason"]


def _parse(content: str, expected: int | None) -> tuple[str, ...]:
    """The numbered lines of `content`, in index order.

    All or nothing: a partial parse would silently shift a TRAKE event onto the
    wrong query, which ranks worse than not rewriting at all. Anything short of
    every index from 1 to `expected` raises and this half falls back.

    `expected=None` is for decomposition, where the count *is* the answer: any
    contiguous run from 1 is accepted, but a gap in the numbering still raises,
    because a missing line there means a dropped event.
    """
    # Belt and braces. The competition endpoint emits no reasoning block with
    # thinking disabled, but a different server or model would.
    body = content.rsplit("</think>", 1)[-1]

    found: dict[int, str] = {}
    for line in body.splitlines():
        match = _LINE.match(line)
        if match:
            found[int(match.group(1))] = match.group(2).strip().strip("\"'“”")

    count = len(found) if expected is None else expected
    if not count or sorted(found) != list(range(1, count + 1)) or not all(found.values()):
        raise ValueError(f"expected {count} rewritten queries, got {sorted(found)}")
    return tuple(found[index] for index in range(1, count + 1))
