"""Read on-screen text and describe each frame, via an OpenAI-compatible VLM.

This fills the 30,750 frames that `join_server_frames.py` could not reach:
short shots, median 2.0 seconds, that this repo's sampling never landed a
keyframe inside. They are the only frames in the target manifest with no text
at all, so they are invisible to every lexical channel.

ONE REQUEST, BOTH FIELDS

Asking separately for the text and the description doubles the request count
against an endpoint already measured to saturate at 8 concurrent requests.
The image is encoded, uploaded and prefilled once either way, so the second
call would pay that cost again to add a sentence.

WHY IT SURVIVES A CLOSED LAPTOP

Every answer is appended and flushed the moment it arrives, and `--resume`
reads back what is already there and skips it. A run killed at 80% restarts
at 80%. That matters more than it sounds: at the measured rate this job is
hours long, the endpoint is shared with the rest of the team, and anything
that has to start over from zero will never finish.

THINKING IS OFF

The served model reasons by default into a separate field, and the first
probe against this endpoint spent its entire token budget there and returned
`content: null`. Reading a ticker needs no deliberation.

    nohup python scripts/vlm_enrich.py --frames data/frames-missing-text.parquet \\
        --out data/vlm-enrich.jsonl --concurrency 8 --resume &
"""

import argparse
import base64
import json
import queue
import sys
import threading
import time
from pathlib import Path
from urllib import error, request

import pyarrow.parquet as pq

PROMPT = """Nhìn ảnh và trả lời đúng hai phần, đúng định dạng dưới đây.

CHỮ:
(chép lại mọi chữ nhìn thấy trên ảnh, mỗi cụm một dòng, giữ nguyên dấu tiếng Việt; nếu ảnh không có chữ nào thì ghi đúng một chữ: KHÔNG)

MÔ TẢ:
(một câu tiếng Việt ngắn gọn tả những gì đang diễn ra trong ảnh)"""

# The model is told to write this when a frame carries no text. Kept as a
# sentinel rather than an empty answer so a refusal and a genuinely blank
# frame stay distinguishable.
NO_TEXT = "KHÔNG"

RETRIES = 3
RETRY_SLEEP_SEC = 5


def encode(path: str) -> str | None:
    try:
        return base64.b64encode(Path(path).read_bytes()).decode("ascii")
    except OSError:
        return None


def parse(answer: str) -> tuple[str, str]:
    """Split the two sections. Anything unparseable yields empty fields.

    A model that ignores the format is a bug to see in the miss count, not
    something to guess the intent of: a wrong caption indexed as if correct
    is worse than no caption.
    """
    text, caption = "", ""
    section = None
    for line in answer.splitlines():
        stripped = line.strip()
        upper = stripped.upper().rstrip(":")
        if upper in ("CHỮ", "CHU"):
            section = "text"
            continue
        if upper in ("MÔ TẢ", "MO TA"):
            section = "caption"
            continue
        if not stripped:
            continue
        if section == "text":
            text = f"{text}\n{stripped}" if text else stripped
        elif section == "caption":
            caption = f"{caption} {stripped}" if caption else stripped

    if text.strip().upper().startswith(NO_TEXT):
        text = ""
    return text.strip(), caption.strip()


def ask(url: str, model: str, image_b64: str, max_tokens: int) -> str | None:
    body = json.dumps(
        {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            },
                        },
                    ],
                }
            ],
        }
    ).encode("utf-8")

    for attempt in range(RETRIES):
        try:
            req = request.Request(
                f"{url}/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with request.urlopen(req, timeout=300) as response:
                data = json.load(response)
            message = data["choices"][0]["message"]
            return (message.get("content") or "").strip()
        except (error.URLError, error.HTTPError, TimeoutError, OSError, KeyError):
            if attempt == RETRIES - 1:
                return None
            time.sleep(RETRY_SLEEP_SEC * (attempt + 1))
    return None


def done_keys(path: Path) -> set[tuple[str, int]]:
    """What a previous run already wrote. Truncated last lines are ignored."""
    if not path.is_file():
        return set()
    keys = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            keys.add((row["video_id"], int(row["keyframe_n"])))
    return keys


def run(args: argparse.Namespace) -> int:
    table = pq.read_table(args.frames)
    rows = list(
        zip(
            table.column("video_id").to_pylist(),
            [int(n) for n in table.column("keyframe_n").to_pylist()],
            table.column("path").to_pylist(),
        )
    )

    out_path = Path(args.out)
    already = done_keys(out_path) if args.resume else set()
    todo = [row for row in rows if (row[0], row[1]) not in already]
    print(f"frames    : {len(rows):,}", flush=True)
    print(f"already   : {len(already):,}", flush=True)
    print(f"to do     : {len(todo):,}", flush=True)
    if not todo:
        return 0

    write_lock = threading.Lock()
    work: queue.Queue = queue.Queue()
    for row in todo:
        work.put(row)

    counts = {"ok": 0, "no_image": 0, "failed": 0, "unparsed": 0}
    started = time.time()

    with open(out_path, "a", encoding="utf-8") as handle:

        def worker() -> None:
            while True:
                try:
                    video_id, keyframe_n, path = work.get_nowait()
                except queue.Empty:
                    return
                image = encode(path)
                if image is None:
                    with write_lock:
                        counts["no_image"] += 1
                    work.task_done()
                    continue

                answer = ask(args.url, args.model, image, args.max_tokens)
                if answer is None:
                    with write_lock:
                        counts["failed"] += 1
                    work.task_done()
                    continue

                text, caption = parse(answer)
                with write_lock:
                    if not text and not caption:
                        counts["unparsed"] += 1
                    counts["ok"] += 1
                    handle.write(
                        json.dumps(
                            {
                                "video_id": video_id,
                                "keyframe_n": keyframe_n,
                                "ocr_text_vlm": text,
                                "caption_vi": caption,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    handle.flush()
                    total = counts["ok"] + counts["failed"] + counts["no_image"]
                    if total % args.log_every == 0:
                        rate = total / max(time.time() - started, 1e-9)
                        left = (len(todo) - total) / max(rate, 1e-9) / 3600
                        print(
                            f"  {total:,}/{len(todo):,}  {rate:.2f}/s  "
                            f"còn ~{left:.1f}h  "
                            f"lỗi={counts['failed']} trống={counts['unparsed']}",
                            flush=True,
                        )
                work.task_done()

        threads = [
            threading.Thread(target=worker, daemon=True)
            for _ in range(args.concurrency)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    elapsed = time.time() - started
    print(
        f"xong: {counts['ok']:,} ghi, {counts['failed']:,} lỗi, "
        f"{counts['no_image']:,} thiếu ảnh, {counts['unparsed']:,} không parse được"
        f"  trong {elapsed / 3600:.2f}h",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--url", default="http://localhost:8002/v1")
    parser.add_argument("--model", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--resume", action="store_true")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
