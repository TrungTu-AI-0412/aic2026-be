# TRAKE retrieval

How `POST /search/trake` turns an overview plus N event descriptions into ranked
frame sequences, and why each stage is shaped the way it is.

Code: [`app/retrieval/tracks.py`](../backend/app/retrieval/tracks.py) (`search_trake`,
`_candidate_videos`, `_best_increasing_sequence`),
[`app/retrieval/engine.py`](../backend/app/retrieval/engine.py)
(`retrieve_video_scores`, `retrieve_per_video`),
[`app/retrieval/rewrite.py`](../backend/app/retrieval/rewrite.py).

---

## 1. What the track has to produce

A TRAKE submission row is `video_id, frame_id_1, ..., frame_id_n` — one frame per
event, in order, inside one video
([`submissions/formats.py`](../backend/app/submissions/formats.py)). That shape
dictates everything downstream:

- A result that fills only 3 of 4 events **is not a row**. Partial sequences are
  dropped, not ranked low.
- The frames must be **strictly increasing**, because the events happen in the
  order the operator typed them.
- The ordering key is `original_frame_id` — what the submission reports, and
  monotone with time inside a video.

## 2. The one idea the design turns on

*Which video* and *where inside it* are different questions.

"Which video" is a whole-video judgement: every event is evidence about the video,
and they accumulate. "Where is event 3" only becomes a well-posed question once the
video is fixed — globally, a fine-grained moment ("the instant all four feet touch
the ground") does not reach the top of a 290k-frame ranking, and never will.

The previous design searched every event globally and intersected the results. It
lost the correct video whenever **any single event** failed to reach a global
top-N, which fine-grained events routinely do. So the path splits in two.

## 3. End-to-end

```mermaid
flowchart TD
    REQ["POST /search/trake\noverview, events[], top_k,\nvideo_ids?, max_gap_sec?"]
    RW["rewrite_queries([overview, *events])\n2 concurrent prompts, 1 batch"]
    REQ --> RW
    RW -->|"vision: EN caption ≤40w\nspeech: VI, narration deleted"| BR{"video_ids given?"}

    BR -->|yes| VID["use them verbatim\n(dedup, order kept)"]
    BR -->|no| A

    subgraph A["STAGE A — choose videos (global)"]
        A1["retrieve_video_scores(overview, 1000)"]
        A2["retrieve_video_scores(event_i, 1000)\nfor every event"]
        A3["score(video) =\nbest_overview + mean_i(best_event_i)"]
        A1 --> A3
        A2 --> A3
        A3 --> A4["top 100 videos"]
    end

    VID --> B
    A4 --> B

    subgraph B["STAGE B — localise (per video)"]
        B1["for each event:\nretrieve_per_video(event, videos, 20)\n= one filtered query PER (video, event)"]
        B1 --> B2["slots[video][event] = 20 candidates\nshots NOT collapsed, rerank OFF"]
    end

    B2 --> C["_best_increasing_sequence\nDP per video"]
    C --> D["drop videos with an empty slot\nor no valid ordering"]
    D --> E["sort by sequence score,\ncut to top_k"]
    E --> F["SearchResult per video:\nframe_ids[] + events[] with alternates"]
```

## 4. Rewrite — one batch, two prompts

`[overview, *events]` goes out as **one numbered batch**, under **two prompts on two
concurrent calls**:

| form | goes to | shape |
|---|---|---|
| `Rewrite.vision` | SigLIP2 text tower + BLIP reranker | English caption, ≤40 words |
| `Rewrite.speech` | ASR transcripts (dense + BM25) | original language, narration **deleted** |

Why batched: every query is searched **twice** — once globally in stage A, once
inside each candidate in stage B. Per-query round trips would put seconds on the
clock for nothing. Measured 1.8s for an overview plus five events.

Why two calls rather than one: merged, the caption rules bled into the deletion
rules and the model variously deleted the subject of the query
(`lễ hội đèn lồng`) or deleted nothing at all. Split, wall clock is the slower of
the two, not their sum, and **each half falls back independently** — a caption that
does not arrive does not cost the cleaned form.

Every failure — box down, timeout, misnumbered line, `finish_reason != "stop"` —
falls back to the query as typed. The `finish_reason` check is load-bearing: a
truncated reply still parses, because the early lines are intact, so without it the
last event is searched as half a sentence.

## 5. Stage A — candidate videos

`_candidate_videos` searches the overview **and every event** globally, then reduces
each to one score per video.

```
score(video) = best_overview_hit(video) + (1/n) · Σ_i best_event_i_hit(video)
```

Three decisions do the work here.

### 5.1 Coverage is deliberately not required

A video the overview alone found still earns a full alignment pass. The events get
another chance *inside* each candidate, so demanding a global hit for each one only
throws away correct videos.

```mermaid
flowchart LR
    subgraph OLD["intersect globally — the old shape"]
        direction TB
        O1["event 1 global top-N"] --> OX(("∩"))
        O2["event 2 global top-N"] --> OX
        O3["event 3 (fine-grained)\n→ misses top-N"] --> OX
        OX --> OR["∅ — correct video lost"]
    end
    subgraph NEW["accumulate evidence — now"]
        direction TB
        N1["overview: 0.31"] --> NX(("+"))
        N2["event 1: 0.28"] --> NX
        N3["event 3: 0.04"] --> NX
        NX --> NR["video still ranks\n→ stage B re-searches it"]
    end
```

### 5.2 Two reductions of the same hit list — not two branches

`retrieve` and `retrieve_video_scores` are **not two paths through one request**.
They are two functions with two different callers, and no request runs both. Both go
through the same `_search_and_boost`: same encode, same 5000-hit Qdrant search, same
clip fusion, same ASR bonus. They differ only in the **last step**.

| function | called by | last step | returns |
|---|---|---|---|
| `retrieve` | `_frame_search` — KIS / QA | `dedupe_by_shot(hits, top_k)` → `[:top_k]` → rerank | `list[ScoredFrame]`, a page of results |
| `retrieve_video_scores` | `_candidate_videos` — Stage A | `max(score)` per `video_id`; no collapse, no cut | `dict[video_id, float]` |
| `retrieve_per_video` | `search_trake` — Stage B | one filtered query per video → sort → `[:20]` | `dict[video_id, list[ScoredFrame]]` |

```mermaid
flowchart BT
    SHARED["_search_and_boost — shared\nencode · 5000 Qdrant hits · clip fusion · ASR bonus"]
    SHARED --> R1["retrieve()\ndedupe_by_shot → [:top_k] → rerank"]
    SHARED --> R2["retrieve_video_scores()\nmax per video_id, no collapse, no cut"]
    SHARED --> R3["retrieve_per_video()\none filtered query per video → [:20]"]
    R1 --> C1["_frame_search — KIS / QA"]
    R2 --> C2["_candidate_videos — Stage A"]
    R3 --> C3["search_trake — Stage B"]
```

### 5.3 Why Stage A cannot use `retrieve`

The lossy step is `[:top_k]`, **not** the shot collapse. Collapsing to one frame per
shot actually *helps* video diversity per slot. What costs videos is that the
surviving slots are a **fixed budget allocated by global score** — and dense hits
concentrate inside a few long videos.

Go deeper in the same ranked list and new videos keep appearing, so any cut discards
every video that first appears below it. Measured on *"close-up of a white lion
head"*:

| depth into the ranked list | distinct videos |
|---|---|
| top 1000 frames | 31 |
| top 5000 frames | 277 |
| top 400 **shots** (what `retrieve` keeps) | 28 |
| **Stage A needs** | **100** |

28 cannot fill a 100-video pool. `retrieve_video_scores` never cuts, so it names
every video present in the 5000 hits. That is also why `dedupe.DEFAULT_CAP` is 5000 —
91ms against 293k points, up from 26ms at 1000.

### 5.4 Reranking is off

BLIP ITM emits probabilities running to 1.0 where cosine scores sit near 0.2. Summed
into a per-video score, the reranked head of each query's list would outweigh every
other video **by scale alone**, so picking candidate videos would come down to
whichever ones BLIP happened to see. It also cost measured seconds per request.

### 5.5 Inside one `retrieve_video_scores` call

```mermaid
flowchart LR
    T["vision text"] --> E["encode_query\nSigLIP2 text tower"]
    E --> S1["frames collection\n5000 hits"]
    E --> S2["clips collection\n5000 hits (if configured)"]
    S1 --> FU["fuse_frames_and_clips\nkey (video_id, shot_id), clip weight 0.5,\nimpute each list's worst score"]
    S2 --> FU
    SP["speech text"] --> SS["asr collection\ndense 0.7 + BM25 0.3, top 200"]
    FU --> BO["apply_asr_bonus\n+0.3 × best segment covering the frame,\nadditive, ±1s pad"]
    SS --> BO
    BO --> MX["max score per video_id"]
```

The ASR bonus is **additive, never multiplicative**: 22 of 873 videos have no
transcript and 4.5% of video time has no segment, so silence is not evidence against
a frame. Speech top-k is pinned at 200 rather than scaled with the 1000-frame pool —
segments are already spread across ~700 of 873 videos in a top-1000 list, so
widening it only cost a second per query for a bonus that reorders near-ties.

## 6. Stage B — localise inside each candidate

For each event, `retrieve_per_video(event.vision, videos, 20, ...)`: one encode, then
**one filtered Qdrant query per video**.

### 6.1 Why not one query filtered to all 100 videos

Because that returns the global top-N *across* them, and starves a correct video that
ranks low overall — the same recall failure the two-stage split exists to fix, one
level down.

```mermaid
flowchart TB
    subgraph ONE["one query, filter = all 100 videos"]
        direction TB
        F1["global top 100 hits"] --> F2["video A: 78 hits\nvideo B: 22 hits\nvideos C…: 0 hits"]
        F2 --> F3["C has an empty slot\n→ dropped, though it may be the answer"]
    end
    subgraph PER["one query per video"]
        direction TB
        G1["100 filtered queries"] --> G2["every video gets its own top 20"]
        G2 --> G3["an empty slot now means\nthe video really has no candidate"]
    end
```

That is what makes the "drop any video with an empty slot" rule safe: after stage B
an empty slot means the video genuinely has nothing for that event, not that the
event lost a global ranking.

### 6.2 Shots are not collapsed here

`dedupe_by_shot` keeps one frame per shot. Two events of a TRAKE query can happen
inside one two-second shot, so collapsing would make that sequence **unrepresentable**
rather than merely rank it worse.

### 6.3 Reranking is off here too

The cross-encoder head covers the top `rerank_top_n` (30) of *one global list*, so
against per-video candidate sets it would rescore an arbitrary slice of them. Making
it meaningful costs one forward pass per (video, event, candidate). The video is
already chosen by here, and within one video the dense space is what separates its
own frames.

### 6.4 One speech query for the whole candidate set

`apply_asr_bonus` min-max normalises the segments it is handed. Boosting each video
from its own segment list would make the bonuses **incomparable between videos** —
which is exactly what ranking the sequences needs. So: one speech query filtered to
all candidates, one bonus pass over the flattened hits, then re-split per video.

## 7. Alignment — `_best_increasing_sequence`

Per video, choose one candidate per event maximising the summed score, subject to two
constraints:

1. `representative_frame` **strictly increasing**
2. `pts_sec` gap between consecutive picks ≤ `max_gap` (default 300s; `0` disables;
   skipped when either timestamp is missing, since a missing time is not evidence
   that the events are far apart)

```mermaid
flowchart LR
    subgraph E1["event 1"]
        A1["f=120\n0.31"]
        A2["f=980\n0.29"]
    end
    subgraph E2["event 2"]
        B1["f=140\n0.26"]
        B2["f=310\n0.24"]
    end
    subgraph E3["event 3"]
        C1["f=100\n0.33"]
        C2["f=450\n0.22"]
    end
    A1 -->|"ok"| B1
    A1 -->|"ok"| B2
    A2 -->|"f decreases ✗"| B1
    B1 -->|"ok"| C2
    B2 -->|"ok"| C2
    B1 -->|"f decreases ✗"| C1
    C1 -->|"unreachable"| X["dropped"]
```

Chosen chain: `120 → 140 → 450`, score `(0.31+0.26+0.22)/3`. Division by the event
count keeps sequences of different lengths on one scale.

It is an **exact** DP, not a greedy walk: whether a candidate may follow another
depends only on that other candidate, so "best total ending at candidate *j*" is a
sufficient state. `max_gap` exists because the events of one TRAKE query describe a
single continuous action — without it a chain spanning a 40-minute video scores
exactly as well as one spanning three seconds.

Ordering is on the frame id, not `pts_sec`, because that is what a submission reports
and it rises with time inside a video anyway. Seconds are used **only** for the gap,
which is a duration and so cannot be expressed in frames when videos differ in frame
rate.

### 7.1 Alternates

`_event_hits` reports, per event, the runners-up the pick beat — held to the *same*
two rules against the **neighbouring picks**: after the previous event, before the
next, within the gap. Offering a candidate the ranker would have rejected would let
an operator hand-assemble a sequence the system does not consider a sequence, and
submit it out of order.

## 8. The knobs

| constant | value | what it buys | when to move it |
|---|---|---|---|
| `TRAKE_VIDEO_CANDIDATES` | 100 | width of the alignment pool; a submission ranks 100 lines and the track returns ≤1 result per candidate | the cost dial — stage B is `videos × events` serial round trips |
| `TRAKE_STAGE_A_TOP_K` | 1000 | frame pool per stage-A query; `overfetch_limit` lifts it to 5000 raw hits | recall of the video pool |
| `TRAKE_CANDIDATES_PER_EVENT` | 20 | candidates per event **inside one video** | when the DP has no valid ordering but the event is visibly present |
| `TRAKE_ALTERNATES_PER_EVENT` | 5 | runners-up offered to the operator | UI only, free |
| `TRAKE_MAX_GAP_SEC` | 300.0 | widest gap between consecutive events | per request via `max_gap_sec`; `0` disables |
| `STAGE_A_SPEECH_TOP_K` | 200 | speech segments matched in stage A | pinned deliberately, see §5.5 |
| `asr_weight` | 0.3 | share of a frame's score speech can add | per request |
| `DEFAULT_CLIP_WEIGHT` | 0.5 | clip index as tie-breaker and recall net, never an outvote | |

## 9. What is off, on purpose

| stage | shot dedupe | rerank | coverage required |
|---|---|---|---|
| Stage A (choose video) | **no** — collapsing names too few videos | **no** — ITM 1.0 vs cosine 0.2 | **no** — loses correct videos |
| Stage B (localise) | **no** — two events can share a shot | **no** — head covers one global list | **yes** — a partial sequence is not a row |
| KIS/QA (`_frame_search`) | yes | yes | n/a |

## 10. Ceilings

- **Latency.** Stage B is serial in both loops: `for event` in `search_trake`, then
  `for video_id` inside `retrieve_per_video`. 100 videos × 5 events = 500 sequential
  Qdrant round trips. `TRAKE_VIDEO_CANDIDATES` is the only dial today.
- **Localisation granularity.** Only as fine as the keyframes — three per shot, so a
  sub-second moment is *bracketed*, not pinned. `events[].alternates` exists so an
  operator closes that gap by hand. Asking the VLM *where* inside a bracketed shot an
  event sits is the obvious next lever.
- **No TRAKE eval set.** `data/eval_set.jsonl` is ASR-derived and has no TRAKE
  queries, so changes here are checked by unit tests and by eye, not measured.
- **Rewrite timeout is flat** (`rewrite_timeout_sec = 6.0`), not `base + n × per_query`
  — a ten-event batch is output-token-bound and can outrun it. Marked `ponytail:` in
  `engine.py`.
- **`retrieve_video_scores` is called `1 + n_events` times**, each pulling 5000 raw
  hits; stage A cost grows linearly with the event count.
