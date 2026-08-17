"""Check third-party shot boundaries against the organiser's own metadata.

Shot lists downloaded from elsewhere are only usable if they were computed on
byte-identical videos. If the organiser re-encoded between batches, frame
indices shift and every `shot_id` silently attaches to the wrong moment — the
same class of failure as reading a keyframe filename as `original_frame_id`.

Verifying against the videos would mean downloading 77 GiB. Two artefacts we
already have pin the frame numbering just as tightly:

- `map-keyframes/<video_id>.csv` gives `frame_idx` and `fps` for every keyframe,
  so keyframes are known-good points that must each land inside some shot.
- `media-info/<video_id>.json` gives `length` in seconds, so the last shot's end
  frame divided by fps must reproduce it.

A shot list that passes both was computed on our exact frame numbering.

Exit code is non-zero when any video fails, so this can gate ingestion.
"""

import argparse
import csv
import json
import statistics
import sys
from bisect import bisect_right
from pathlib import Path

# A shot list may legitimately stop a beat before the final frame (trailing
# black frames, encoder padding), and `media-info.length` is rounded to whole
# seconds. These bound how much of that slack is tolerated.
MAX_DURATION_DRIFT_SEC = 2.0
MAX_TAIL_GAP_SEC = 2.0


def load_keyframes(path: Path) -> tuple[list[int], float]:
    """Return (sorted frame_idx values, fps) from a map-keyframes CSV."""
    frames: list[int] = []
    fps_values: list[float] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            frames.append(int(float(row["frame_idx"])))
            fps_values.append(float(row["fps"]))
    if not frames:
        raise ValueError("empty map-keyframes CSV")
    return sorted(frames), statistics.median(fps_values)


def load_shots(path: Path) -> list[tuple[int, int]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    shots = [(int(a), int(b)) for a, b in raw]
    if not shots:
        raise ValueError("empty shot list")
    return shots


def check_video(
    video_id: str,
    shots: list[tuple[int, int]],
    frames: list[int],
    fps: float,
    length_sec: float | None,
) -> dict:
    problems: list[str] = []

    ordered = all(a <= b for a, b in shots)
    monotonic = all(
        shots[i][1] < shots[i + 1][0] for i in range(len(shots) - 1)
    )
    if not ordered:
        problems.append("shot with end < start")
    if not monotonic:
        problems.append("shots overlap or are unsorted")

    # Keyframe containment. Shots are sorted and disjoint, so a binary search
    # over start frames finds the only candidate that can contain a frame.
    starts = [s for s, _ in shots]
    uncovered = 0
    for frame in frames:
        index = bisect_right(starts, frame) - 1
        if index < 0 or frame > shots[index][1]:
            uncovered += 1
    if uncovered:
        problems.append(f"{uncovered}/{len(frames)} keyframes outside any shot")

    # Range: the shot list must reach at least as far as the last keyframe.
    last_shot_end = shots[-1][1]
    tail_gap_frames = frames[-1] - last_shot_end
    if tail_gap_frames > 0:
        problems.append(
            f"last keyframe {frames[-1]} is {tail_gap_frames} frames past "
            f"last shot end {last_shot_end}"
        )

    # Duration: independent confirmation that fps and frame numbering agree.
    drift = None
    if length_sec:
        shot_seconds = (last_shot_end + 1) / fps
        drift = shot_seconds - length_sec
        if abs(drift) > MAX_DURATION_DRIFT_SEC:
            problems.append(
                f"duration drift {drift:+.1f}s "
                f"(shots {shot_seconds:.1f}s vs media-info {length_sec:.1f}s)"
            )

    return {
        "video_id": video_id,
        "shots": len(shots),
        "keyframes": len(frames),
        "fps": fps,
        "uncovered": uncovered,
        "tail_gap_frames": tail_gap_frames,
        "drift_sec": drift,
        "problems": problems,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=Path, required=True)
    parser.add_argument("--map-keyframes", type=Path, required=True)
    parser.add_argument("--media-info", type=Path, required=True)
    parser.add_argument(
        "--show", type=int, default=15, help="how many failures to print"
    )
    args = parser.parse_args(argv)

    shot_files = {
        # annotations/shot_json/L21/V001.json -> L21_V001
        f"{path.parent.name}_{path.stem}": path
        for path in args.shots.rglob("*.json")
        if path.parent.name.startswith(("L", "K"))
    }
    keyframe_files = {p.stem: p for p in args.map_keyframes.glob("*.csv")}
    media_files = {p.stem: p for p in args.media_info.glob("*.json")}

    print(f"shot lists     : {len(shot_files)}")
    print(f"map-keyframes  : {len(keyframe_files)}")
    print(f"media-info     : {len(media_files)}")

    missing = sorted(set(keyframe_files) - set(shot_files))
    extra = sorted(set(shot_files) - set(keyframe_files))
    print(f"no shot list   : {len(missing)}  {missing[:5]}")
    print(f"unmatched shots: {len(extra)}  {extra[:5]}")

    results = []
    errors = []
    for video_id in sorted(set(shot_files) & set(keyframe_files)):
        try:
            frames, fps = load_keyframes(keyframe_files[video_id])
            shots = load_shots(shot_files[video_id])
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            errors.append(f"{video_id}: unreadable ({exc})")
            continue

        length = None
        if video_id in media_files:
            try:
                length = float(
                    json.loads(
                        media_files[video_id].read_text(encoding="utf-8")
                    ).get("length")
                    or 0
                )
            except (ValueError, TypeError, OSError, json.JSONDecodeError):
                length = None

        results.append(check_video(video_id, shots, frames, fps, length or None))

    checked = len(results)
    failed = [r for r in results if r["problems"]]
    print(f"\nchecked        : {checked}")
    print(f"passed         : {checked - len(failed)}")
    print(f"failed         : {len(failed)}")
    if errors:
        print(f"unreadable     : {len(errors)}")

    if checked:
        total_shots = sum(r["shots"] for r in results)
        total_frames = sum(r["keyframes"] for r in results)
        uncovered = sum(r["uncovered"] for r in results)
        drifts = [abs(r["drift_sec"]) for r in results if r["drift_sec"] is not None]
        print(f"\ntotal shots    : {total_shots:,}")
        print(f"total keyframes: {total_frames:,}")
        print(
            f"keyframes outside a shot: {uncovered:,} "
            f"({uncovered / total_frames * 100:.3f}%)"
        )
        print(f"shots per video: median {statistics.median(r['shots'] for r in results):.0f}")
        if drifts:
            drifts.sort()
            print(
                f"duration drift : median {statistics.median(drifts):.2f}s  "
                f"p95 {drifts[int(len(drifts) * 0.95)]:.2f}s  "
                f"max {drifts[-1]:.2f}s"
            )

    for result in failed[: args.show]:
        print(f"\nFAIL {result['video_id']}")
        for problem in result["problems"]:
            print(f"     {problem}")
    if len(failed) > args.show:
        print(f"\n... and {len(failed) - args.show} more failures")
    for line in errors[:5]:
        print(f"ERROR {line}")

    return 1 if failed or errors or missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
