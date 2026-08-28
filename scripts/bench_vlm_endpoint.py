"""Measure what an OpenAI-compatible VLM endpoint actually does per second.

Throughput here is not a property of the model alone: it depends on how many
requests are in flight, because vLLM batches whatever has arrived. A serial
loop measures latency and calls it throughput, which is how a job gets
estimated at twenty minutes and takes six hours.

THINKING IS OFF

The served model reasons by default and writes that reasoning into a separate
field, so a request can burn its whole token budget before emitting one
character of answer — the first probe against this endpoint returned
`content: null` with a paragraph of deliberation in `reasoning`. For reading a
ticker off a frame there is nothing to deliberate about, so it is disabled and
the budget goes to the answer.

    python scripts/bench_vlm_endpoint.py --images DIR --concurrency 8 16
"""

import argparse
import base64
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib import error, request

OCR_PROMPT = (
    "Chép lại chính xác mọi chữ xuất hiện trên ảnh, mỗi dòng một cụm. "
    "Không mô tả, không giải thích. Nếu ảnh không có chữ, trả lời: KHÔNG CÓ."
)
CAPTION_PROMPT = (
    "Mô tả ngắn gọn bằng tiếng Việt những gì diễn ra trong ảnh, một câu."
)


def encode(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def ask(url: str, model: str, image_b64: str, prompt: str, max_tokens: int) -> tuple[str, int]:
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        # vLLM passes this through to the chat template; Qwen3 reads it to
        # skip the reasoning block entirely.
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }
        ],
    }
    payload = json.dumps(body).encode("utf-8")
    req = request.Request(
        f"{url}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with request.urlopen(req, timeout=300) as response:
            data = json.load(response)
    except error.HTTPError as exc:
        return f"HTTP {exc.code}: {exc.read()[:200].decode('utf-8', 'replace')}", 0
    message = data["choices"][0]["message"]
    text = message.get("content") or message.get("reasoning") or ""
    used = data.get("usage", {}).get("completion_tokens", 0)
    return text.strip(), used


def run(args: argparse.Namespace) -> int:
    images = sorted(Path(args.images).rglob("*.jpg"))[: args.count]
    if not images:
        print(f"error: no .jpg under {args.images}", file=sys.stderr)
        return 1
    encoded = [encode(path) for path in images]
    prompt = CAPTION_PROMPT if args.task == "caption" else OCR_PROMPT
    print(f"images   : {len(images)}  task={args.task}")

    sample, _ = ask(args.url, args.model, encoded[0], prompt, args.max_tokens)
    print(f"sample   : {sample[:200]!r}\n")

    print(f"{'conc':>5} {'img/s':>8} {'tok/s':>9} {'s/ảnh':>8}   ước tính cho")
    print(f"{'':>5} {'':>8} {'':>9} {'':>8}   30.750 / 289.881")
    for concurrency in args.concurrency:
        started = time.time()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(
                pool.map(
                    lambda b: ask(args.url, args.model, b, prompt, args.max_tokens),
                    encoded,
                )
            )
        elapsed = time.time() - started
        rate = len(images) / elapsed
        tokens = sum(used for _, used in results) / elapsed
        small = 30_750 / rate / 3600
        full = 289_881 / rate / 3600
        print(
            f"{concurrency:>5} {rate:>8.2f} {tokens:>9.0f} {1 / rate:>8.2f}   "
            f"{small:>5.1f}h / {full:.1f}h"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8002/v1")
    parser.add_argument("--model", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--images", required=True)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--task", choices=("ocr", "caption"), default="ocr")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 16])
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
