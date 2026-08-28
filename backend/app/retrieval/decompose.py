"""Turning one pasted query into the overview and events a temporal search needs.

An operator pastes the task exactly as the competition wrote it and reviews
what came back before any searching happens. That review is the point: a wrong
decomposition is a whole task lost, and it is only visible if someone can see
it next to the words it was cut from.

Two paths, chosen by the text itself:

- **Markers.** Real TRAKE tasks enumerate their own moments (`E1:`, `E2:` ...),
  so the split is a regex, not a model. That is not just cheaper - it is the
  only version that cannot renumber or merge them. One of the three TRAKE
  queries in `data/evaluation_set_p1.csv` is numbered `E1, E2, E2, E4`; it has
  four events, and splitting on the marker *lines* says so while trusting the
  numbers does not.
- **Prose.** A KIS description walks through phases in sentences ("phân cảnh
  tiếp theo...", "sau đó...", "bước đầu tiên..."). Nothing to parse, so the
  model segments it, capped, because stage B pays `videos x events` round trips
  for every extra event.

Both paths end at the same caption call, and the event lines are captioned
*together with the overview* on purpose: "khoảnh khắc 4 chân hoàn toàn chạm
đất" is searched alone against single frames, and alone it describes nothing -
it needs the lion, its colours and the competition floor that the overview
states once and every frame of that scene shows. That is why this path does not
reuse `rewrite.CAPTION_PROMPT`, which captions each line in isolation.

What differs is where the *speech* form comes from, and that decides what can
run beside what:

- Markers give the events already split, but a marker query still opens with
  narration ("..., tìm các sự kiện sau:"), so the speech form is
  `rewrite.CLEAN_PROMPT` - and with nothing to wait for, it runs **beside** the
  caption call, at the wall clock of the slower one.
- Prose has to be split before anything can be captioned or cleaned, and the
  split *is* the deletion (each event is a span of the operator's own words),
  so decomposition produces the speech form itself and the caption call runs
  **after** it.

The prose order is serial on purpose. Two calls that each decompose
independently can return different event counts and different boundaries, and
retrieval pairs the two forms positionally - a silent off-by-one that searches
event 3's caption against event 2's speech. One call owning the split makes
that unrepresentable, and it costs a second on a screen where a human is
reading anyway.

Failure is not uniform here, because the two calls are not worth the same. No
decomposition is fatal: it is the entire product of the request, and the
operator falls back to typing events into the form by hand. A failed caption
call is not: the Vietnamese decomposition is in hand, and `vision=None` means
retrieval puts the original-language text through the image tower - worse than
a caption, and much better than nothing.
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from app.retrieval.rewrite import (
    CLEAN_PROMPT,
    DEFAULT_CAPTION_WORDS,
    TOKENS_PER_CAPTION_WORD,
    _attempt,
    _call,
    _clean_budget,
    caption_word_cap,
)
from app.schemas.search import DecomposeResponse, QueryForms

if TYPE_CHECKING:  # pragma: no cover - import cycle: engine imports rewrite
    from app.retrieval.engine import RetrievalConfig, Timings


class DecompositionUnavailableError(Exception):
    """The query could not be decomposed at all, so there is nothing to review."""


def _decompose_prompt(max_events: int) -> str:
    return (
        "You split a video-search query into the moments it walks through.\n"
        "The query is an operator's description of something that happens over"
        " time, wrapped in narration about the video or the search itself"
        ' ("đoạn video mô tả", "phân cảnh tiếp theo", "hãy tìm").\n'
        "This is a SPLITTING and DELETION task, not a writing task. Keep the"
        " language the query arrived in and the operator's own words: never"
        " translate, paraphrase, reorder, summarise or respell anything you"
        " keep. Remove only the words that refer to the video, the clip, the"
        " scene, the frame or the act of searching.\n"
        "Output numbered lines.\n"
        "Line 1 is the overview: what the whole query is about, in one short"
        " phrase. Copy it from the query when the query states one; when it does"
        " not, build one from the query's own words naming the subject and the"
        " setting, and nothing else.\n"
        "Every later line is one moment, in the order they happen. A moment is"
        " something a single frame can show: a state, a contact, an appearance,"
        " the start or the end of an action. Split where the query itself moves"
        ' on ("phân cảnh tiếp theo", "sau đó", "bước đầu tiên", "cuối cùng").\n'
        f"At most {max_events} moment lines. Merge the least distinctive moments"
        " if the query walks through more. Never output a moment the query does"
        " not describe.\n"
        "Nothing else: no preamble, no blank lines, no commentary."
    )


def event_caption_prompt(word_cap: int = DEFAULT_CAPTION_WORDS) -> str:
    """The same cap `rewrite.caption_prompt` uses, for the same reason.

    An event is searched alone against single frames through the same text
    tower, so it has the same window to fit into. Hard-coding 40 here would
    quietly re-impose SigLIP2's budget on a profile that does not share it.
    """
    return (
    "You turn the parts of a decomposed video-search query into captions for an"
    " image search over single frames.\n"
    "Line 1 is the overview of the whole scene. Every later line is one moment"
    " inside that scene, and each one is searched ALONE against single frames -"
    " so carry the scene's lasting visual detail into every caption: the"
    " subjects, how many, their colours and clothing, the objects, the setting."
    " The moment's own action stays the subject of its caption.\n"
    f"English, AT MOST {word_cap} words per line - count them, a longer caption is cut"
    " off and wasted. Spend those words on what a camera records: the camera"
    " angle or shot type when the input states one (overhead, top-down,"
    " head-on, close-up, wide), how many of each thing, colours, clothing,"
    " objects, where things sit relative to each other, the setting,"
    " expressions, the action.\n"
    "Add only detail stated somewhere in the input, and never invent one: if"
    " nothing mentions the camera, do not mention the camera; if nothing gives a"
    " colour, do not give one. Leave out only what no frame can show: the"
    " narration about the video and the search itself. Never merge or split"
    " lines.\n"
    "Output one line per input: the same number, a period, a space, then the"
    " caption. Nothing else."
    )


EVENT_CAPTION_PROMPT = event_caption_prompt()

# `E1:`, `e 2.`, at the start of the query or after any whitespace. Anchored on
# whitespace rather than on line starts because a task pasted out of a PDF
# arrives on one line; the trailing separator is what keeps it off ordinary
# prose, where a bare "E2" is not followed by a colon.
_MARKER = re.compile(r"(?:\A|(?<=\s))E\s*\d+\s*[:.]\s*", re.IGNORECASE)


def split_markers(query: str) -> tuple[str, list[str]] | None:
    """`(overview, events)` when the query enumerates its own events, else None.

    The overview is whatever precedes the first marker, empty when the task
    opens straight into `E1:`. Events are counted by marker, never by the
    number inside it.
    """
    markers = list(_MARKER.finditer(query))
    if len(markers) < 2:
        return None

    starts = [match.end() for match in markers]
    ends = [match.start() for match in markers[1:]] + [len(query)]
    events = [query[start:end].strip() for start, end in zip(starts, ends)]
    if not all(events):
        return None
    return query[: markers[0].start()].strip(), events


def decompose(
    query: str, max_events: int, config: "RetrievalConfig", timings: "Timings"
) -> DecomposeResponse:
    """The overview and events of `query`, in both retrieval forms."""
    started = time.perf_counter()
    markers = split_markers(query)
    try:
        if markers is not None:
            head, events = markers
        else:
            head, events = _decompose_prose(query, max_events, config)
    finally:
        timings.record("decompose", started)

    # A task that opens straight into `E1:` states no overview. The events' own
    # words are then the only honest one: every term the operator typed, which
    # is what stage A wants against speech, and enough for the caption call to
    # write a scene-level line from. `original` stays None either way, so the
    # review screen shows an overview nobody typed as exactly that.
    parts = [head or " ".join(events), *events]

    if markers is not None:
        # Nothing to wait for: the split is already done, so the two forms are
        # produced side by side and the step costs the slower call, not both.
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = pool.submit(_captions, parts, config, timings)
            speech = _attempt(CLEAN_PROMPT, tuple(parts), config, _clean_budget(tuple(parts)))
            captions = pending.result()
        # Deletion is an improvement, never a dependency: the marker text as
        # typed still searches, it just carries "tìm các sự kiện sau" with it.
        speech = list(speech or parts)
    else:
        speech = parts
        captions = _captions(parts, config, timings)

    forms = [
        QueryForms(
            original=original,
            vision=captions[index] if captions else None,
            speech=speech[index],
        )
        for index, original in enumerate(
            # Prose events are the model's split of the operator's sentences, so
            # there is no span to point at - the screen shows them under the
            # query they came from. Marker events have one exactly.
            [head or None, *(events if markers is not None else [None] * len(events))]
        )
    ]
    return DecomposeResponse(
        source="markers" if markers is not None else "llm",
        overview=forms[0],
        events=forms[1:],
        latency_ms=timings.as_dict(),
    )


def _decompose_prose(
    query: str, max_events: int, config: "RetrievalConfig"
) -> tuple[str, list[str]]:
    """Segment a query that does not enumerate itself, or raise.

    Raising rather than degrading is the whole difference between this call and
    every other LLM call on the query path: a 200 carrying a decomposition
    nobody performed is the failure you cannot see on a screen, in a room where
    you have three minutes.
    """
    if not config.rewrite_enabled or not config.rewrite_base_url:
        raise DecompositionUnavailableError("query decomposition is not configured")

    try:
        lines = _call(
            _decompose_prompt(max_events),
            (query,),
            # The output is the query re-emitted in pieces, so it is sized from
            # the query, not from the event count. Vietnamese runs about two
            # characters per token in this tokenizer.
            len(query) // 2 + 32 * (max_events + 1) + 64,
            None,
            config.rewrite_base_url,
            config.rewrite_model,
            config.rewrite_api_key,
            config.rewrite_timeout_sec,
        )
    except Exception as exc:
        raise DecompositionUnavailableError(f"query decomposition failed: {exc}") from exc

    overview, *events = lines
    if not events:
        raise DecompositionUnavailableError("decomposition returned no events")
    # Over the cap is a failed call, not something to truncate: the tail of
    # these descriptions is where the distinguishing detail sits, so dropping
    # it silently is worse than making the operator press the button again.
    if len(events) > max_events:
        raise DecompositionUnavailableError(
            f"decomposition returned {len(events)} events, over the cap of {max_events}"
        )
    return overview, events


def _captions(
    parts: list[str], config: "RetrievalConfig", timings: "Timings"
) -> tuple[str, ...] | None:
    """English captions for the overview and every event, or None if none came."""
    if not config.rewrite_enabled or not config.rewrite_base_url:
        return None

    started = time.perf_counter()
    try:
        word_cap = caption_word_cap(config)
        return _call(
            event_caption_prompt(word_cap),
            tuple(parts),
            int(word_cap * TOKENS_PER_CAPTION_WORD) * len(parts) + 64,
            len(parts),
            config.rewrite_base_url,
            config.rewrite_model,
            config.rewrite_api_key,
            config.rewrite_timeout_sec,
        )
    except Exception:
        return None
    finally:
        timings.record("caption", started)
