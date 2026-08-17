"""Scrape YouTube transcripts for every video referenced by `media-info/`.

This is a data-acquisition tool, intentionally separate from the ingestion
package: it is the one step that needs the public internet, and the competition
query path must stay offline. It runs once, offline of the competition, and
writes JSON that later feeds a Parquet manifest.

The organiser ships `media-info/<video_id>.json` containing a `watch_url`, so a
scraped transcript is free ASR: same speech, already segmented, already
punctuated, with none of the GPU cost or word-error rate of running Whisper over
873 videos. Videos whose captions are disabled still need ASR — `--report` lists
them.

Output is one JSON per video under `--out`, keyed by the dataset `video_id`
(*not* the YouTube id) so downstream joins never have to re-parse a URL:

    {"video_id", "youtube_id", "language", "language_code", "is_generated",
     "duration_sec", "segments": [{"start", "duration", "text"}, ...]}

Reruns skip videos already written, so an interrupted run resumes by repeating
the same command.
"""

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    from youtube_transcript_api import (
        IpBlocked,
        RequestBlocked,
        YouTubeTranscriptApi,
    )
    from youtube_transcript_api._errors import CouldNotRetrieveTranscript
    from youtube_transcript_api.proxies import (
        GenericProxyConfig,
        WebshareProxyConfig,
    )
except ImportError:  # pragma: no cover - dependency is install-time, not runtime
    sys.exit("missing dependency: pip install youtube-transcript-api")


# Vietnamese first: the corpus is Vietnamese news. English is kept as a
# fallback because a handful of clips carry only auto-translated captions, and
# a wrong-language transcript still beats no transcript for entity matching.
DEFAULT_LANGUAGES = ("vi", "vi-VN", "en", "en-US")

_YOUTUBE_ID = re.compile(
    r"(?:v=|/videos/|/embed/|youtu\.be/|/shorts/)([0-9A-Za-z_-]{11})"
)

_print_lock = threading.Lock()


class ScrapeError(RuntimeError):
    pass


class BlockedError(ScrapeError):
    """YouTube is rejecting this IP outright.

    Distinct from a per-video failure: once it fires, every subsequent request
    fails too, so the run must stop instead of marching through the remaining
    videos and recording hundreds of meaningless failures.
    """


def _log(message: str) -> None:
    with _print_lock:
        print(message, flush=True)


def extract_youtube_id(record: dict) -> str:
    """Pull the 11-character video id out of a media-info record.

    `watch_url` is the documented field, but a few records only carry the id in
    `thumbnail_url` or `channel_url`-adjacent fields, so every string value is
    scanned before giving up.
    """
    url = record.get("watch_url")
    if isinstance(url, str):
        match = _YOUTUBE_ID.search(url)
        if match:
            return match.group(1)

    for value in record.values():
        if isinstance(value, str):
            match = _YOUTUBE_ID.search(value)
            if match:
                return match.group(1)

    raise ScrapeError("no YouTube id found in media-info record")


def load_targets(media_info_dir: Path) -> list[tuple[str, str]]:
    """Return (video_id, youtube_id) for every readable media-info file."""
    files = sorted(media_info_dir.glob("*.json"))
    if not files:
        raise ScrapeError(f"no *.json files under {media_info_dir}")

    targets: list[tuple[str, str]] = []
    for path in files:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _log(f"SKIP  {path.name}: unreadable ({exc})")
            continue
        try:
            targets.append((path.stem, extract_youtube_id(record)))
        except ScrapeError as exc:
            _log(f"SKIP  {path.name}: {exc}")
    return targets


def _pick_transcript(transcript_list, languages: tuple[str, ...]):
    """Prefer a human transcript, fall back to auto-generated, then anything.

    Manually written captions are punctuated and correctly spell proper nouns,
    which is exactly what sparse text search keys on, so they are worth
    preferring even when an auto-generated track exists in the same language.
    """
    try:
        return transcript_list.find_manually_created_transcript(list(languages))
    except Exception:  # noqa: BLE001 - library raises several unrelated types
        pass
    try:
        return transcript_list.find_generated_transcript(list(languages))
    except Exception:  # noqa: BLE001
        pass
    for transcript in transcript_list:
        return transcript
    raise ScrapeError("transcript list is empty")


def fetch_one(
    api: YouTubeTranscriptApi,
    video_id: str,
    youtube_id: str,
    languages: tuple[str, ...],
    retries: int,
) -> dict:
    last_error: Exception | None = None

    for attempt in range(retries + 1):
        try:
            transcript_list = api.list(youtube_id)
            transcript = _pick_transcript(transcript_list, languages)
            fetched = transcript.fetch()
            segments = [
                {
                    "start": round(float(snippet.start), 3),
                    "duration": round(float(snippet.duration), 3),
                    "text": snippet.text.strip(),
                }
                for snippet in fetched
                if snippet.text and snippet.text.strip()
            ]
            duration = (
                round(segments[-1]["start"] + segments[-1]["duration"], 3)
                if segments
                else 0.0
            )
            return {
                "video_id": video_id,
                "youtube_id": youtube_id,
                "language": transcript.language,
                "language_code": transcript.language_code,
                "is_generated": bool(transcript.is_generated),
                "duration_sec": duration,
                "segments": segments,
            }
        except (IpBlocked, RequestBlocked) as exc:
            raise BlockedError(type(exc).__name__) from exc
        except CouldNotRetrieveTranscript as exc:
            # Captions disabled / video removed / age-gated: retrying cannot
            # help, so fail fast and let the caller record it for ASR.
            raise ScrapeError(type(exc).__name__) from exc
        except Exception as exc:  # noqa: BLE001 - transport errors vary widely
            last_error = exc
            if attempt < retries:
                # Exponential backoff with jitter: YouTube throttles bursts per
                # IP, and a synchronised retry storm across threads is what
                # turns throttling into a hard block.
                time.sleep((2**attempt) + random.uniform(0, 1.0))

    raise ScrapeError(f"{type(last_error).__name__}: {last_error}")


def scrape(
    targets: list[tuple[str, str]],
    out_dir: Path,
    languages: tuple[str, ...],
    workers: int,
    retries: int,
    delay: float,
    cooldown: float,
    max_cooldowns: int,
    proxy_config=None,
) -> tuple[int, int, list[dict]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    api = YouTubeTranscriptApi(proxy_config=proxy_config)

    pending = [(v, y) for v, y in targets if not (out_dir / f"{v}.json").exists()]
    done_already = len(targets) - len(pending)
    if done_already:
        _log(f"resume: {done_already} transcript(s) already on disk")

    counters = {"ok": 0, "fail": 0, "n": 0, "cooldowns": 0}
    failures: list[dict] = []
    lock = threading.Lock()
    cooldown_lock = threading.Lock()
    blocked = threading.Event()
    total = len(pending)

    def serve_cooldown() -> bool:
        """Sleep off an IP block. Returns False once the budget is spent.

        The block is per-IP and time-based, so the only cure is to stop asking
        for a while. Holding the lock across the sleep means a concurrent
        worker that also tripped the block waits here rather than starting a
        second, redundant cooldown.
        """
        with cooldown_lock:
            if blocked.is_set():
                return False
            if counters["cooldowns"] >= max_cooldowns:
                blocked.set()
                _log(
                    f"\nSTOP: still blocked after {max_cooldowns} cooldown(s). "
                    f"{counters['ok']} transcript(s) saved this run.\n"
                    "      Rerun the same command later to resume, or switch "
                    "that remainder to Whisper."
                )
                return False
            counters["cooldowns"] += 1
            index = counters["cooldowns"]
        _log(
            f"\nBLOCKED by YouTube — cooldown {index}/{max_cooldowns}, "
            f"sleeping {cooldown / 60:.0f} min "
            f"({counters['ok']} saved so far)\n"
        )
        time.sleep(cooldown)
        return True

    def work(item: tuple[str, str]) -> None:
        video_id, youtube_id = item
        while True:
            if blocked.is_set():
                return
            if delay:
                time.sleep(random.uniform(0, delay))
            try:
                payload = fetch_one(api, video_id, youtube_id, languages, retries)
                break
            except BlockedError:
                if serve_cooldown():
                    continue  # same video, after the wait
                return
            except ScrapeError as exc:
                with lock:
                    counters["fail"] += 1
                    counters["n"] += 1
                    index = counters["n"]
                    failures.append(
                        {
                            "video_id": video_id,
                            "youtube_id": youtube_id,
                            "reason": str(exc),
                        }
                    )
                _log(f"[{index}/{total}] FAIL {video_id} ({youtube_id}): {exc}")
                return

        tmp = out_dir / f".{video_id}.json.tmp"
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        tmp.replace(out_dir / f"{video_id}.json")

        with lock:
            counters["ok"] += 1
            counters["n"] += 1
            index = counters["n"]
        kind = "auto" if payload["is_generated"] else "manual"
        _log(
            f"[{index}/{total}] OK   {video_id}  {payload['language_code']}/{kind}  "
            f"{len(payload['segments'])} seg  {payload['duration_sec']:.0f}s"
        )

    if pending:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(work, pending))

    if blocked.is_set():
        remaining = total - counters["ok"] - counters["fail"]
        _log(f"{remaining} video(s) left untouched by the block")

    return counters["ok"], counters["fail"], failures


def report(out_dir: Path, targets: list[tuple[str, str]]) -> None:
    languages: dict[str, int] = {}
    generated = manual = empty = 0
    segments_total = 0
    seconds_total = 0.0
    present: set[str] = set()

    for path in sorted(out_dir.glob("*.json")):
        if path.name.startswith("_"):  # bookkeeping, e.g. _failures.json
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        present.add(payload["video_id"])
        code = payload.get("language_code", "?")
        languages[code] = languages.get(code, 0) + 1
        if payload.get("is_generated"):
            generated += 1
        else:
            manual += 1
        count = len(payload.get("segments", []))
        segments_total += count
        seconds_total += float(payload.get("duration_sec", 0.0))
        if count == 0:
            empty += 1

    print(f"\ntranscripts on disk : {len(present)}")
    print(f"  manual / auto     : {manual} / {generated}")
    print(f"  empty (0 segments): {empty}")
    print(f"  total segments    : {segments_total:,}")
    print(f"  total speech      : {seconds_total / 3600:.1f} h")
    print(f"  languages         : {dict(sorted(languages.items()))}")

    missing = [v for v, _ in targets if v not in present]
    print(f"\nmissing (need ASR)  : {len(missing)}")
    for video_id in missing[:20]:
        print(f"  {video_id}")
    if len(missing) > 20:
        print(f"  ... and {len(missing) - 20} more")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--media-info",
        type=Path,
        required=True,
        help="directory of <video_id>.json files shipped by the organiser",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output directory for <video_id>.json transcripts",
    )
    parser.add_argument(
        "--languages",
        default=",".join(DEFAULT_LANGUAGES),
        help="comma-separated language preference order",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="max random pre-request sleep in seconds, to desynchronise workers",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=900.0,
        help="seconds to pause after YouTube blocks the IP",
    )
    parser.add_argument(
        "--max-cooldowns",
        type=int,
        default=8,
        help="give up after this many blocks in one run",
    )
    parser.add_argument(
        "--webshare",
        action="store_true",
        help=(
            "route requests through Webshare residential proxies; reads "
            "WEBSHARE_USERNAME and WEBSHARE_PASSWORD from the environment"
        ),
    )
    parser.add_argument(
        "--proxy",
        help=(
            "route requests through this HTTP proxy URL, e.g. "
            "http://user:pass@host:port (used when --webshare is not set)"
        ),
    )
    parser.add_argument(
        "--shard",
        help=(
            "take only this slice of the videos, as I/N (e.g. 2/5). "
            "The rate limit is per IP, so N teammates on N home connections "
            "each run a different shard and the work finishes in one pass."
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="stop after N videos (0 = all)"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="summarise what is on disk and exit without scraping",
    )
    args = parser.parse_args(argv)

    try:
        targets = load_targets(args.media_info)
    except ScrapeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"media-info records: {len(targets)}")

    if args.report:
        report(args.out, targets)
        return 0

    if args.shard:
        try:
            index, total = (int(part) for part in args.shard.split("/", 1))
        except ValueError:
            print(f"error: --shard must look like 2/5, got {args.shard!r}", file=sys.stderr)
            return 2
        if not 1 <= index <= total:
            print(f"error: --shard {args.shard} is out of range", file=sys.stderr)
            return 2
        # Stride rather than contiguous blocks: videos are ordered by lot, so
        # a contiguous slice would give one person all of L26 (498 videos) and
        # another almost nothing.
        targets = targets[index - 1 :: total]
        print(f"shard {index}/{total}: {len(targets)} videos")

    if args.limit:
        targets = targets[: args.limit]

    languages = tuple(x.strip() for x in args.languages.split(",") if x.strip())

    # YouTube rate-limits per IP, and one home connection runs dry after ~64
    # videos. A proxy pool is the only way to finish in a single pass from one
    # machine; without one, split the work with --shard across teammates.
    proxy_config = None
    if args.webshare:
        username = os.environ.get("WEBSHARE_USERNAME")
        password = os.environ.get("WEBSHARE_PASSWORD")
        if not username or not password:
            print(
                "error: --webshare needs WEBSHARE_USERNAME and "
                "WEBSHARE_PASSWORD in the environment",
                file=sys.stderr,
            )
            return 2
        proxy_config = WebshareProxyConfig(
            proxy_username=username, proxy_password=password
        )
        print("proxy: Webshare residential")
    elif args.proxy:
        proxy_config = GenericProxyConfig(
            http_url=args.proxy, https_url=args.proxy
        )
        print("proxy: generic")

    started = time.monotonic()
    ok, fail, failures = scrape(
        targets,
        args.out,
        languages,
        args.workers,
        args.retries,
        args.delay,
        args.cooldown,
        args.max_cooldowns,
        proxy_config,
    )
    elapsed = time.monotonic() - started

    if failures:
        failure_path = args.out / "_failures.json"
        failure_path.write_text(
            json.dumps(failures, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"\nfailures written to {failure_path}")

    print(f"\nOK={ok} FAIL={fail} in {elapsed / 60:.1f} min")
    report(args.out, targets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
