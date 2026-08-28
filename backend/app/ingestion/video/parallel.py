"""Run a per-video stage across worker processes.

Video decode is the bottleneck of every stage in this package, and on this
machine it is CPU-bound: four cores against 12.3M frames. The GPU work that
follows (a 7.6M-parameter shot detector, or a resize) is comparatively free.

Processes rather than threads, for two reasons. PyAV's log callback takes the
GIL from whatever thread FFmpeg logs on, which deadlocks against a main thread
already holding it inside container teardown -- `app/features/media.py` has the
full account, and it is why decode stays single-threaded *within* a worker.
Processes have no shared GIL, so they sidestep that entirely, and 3 workers at
428 fps each beat the 705 fps that threaded decode measured.

The spawn context is deliberate: forking a process that has already initialised
CUDA gives the child a broken context, and these workers each load a model.
"""

from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os
from typing import Any

from tqdm import tqdm

# Three, not four: one core stays free for the parent process, the progress bar
# and the Parquet flushes. Raising this past the core count only adds context
# switching, since every worker is already CPU-saturated.
DEFAULT_WORKERS = 3


def resolve_workers(requested: int | None) -> int:
    """Clamp a requested worker count to something this machine can run."""
    if requested is not None and requested > 0:
        return requested
    cores = os.cpu_count() or 1
    return max(1, min(DEFAULT_WORKERS, cores - 1))


def map_videos(
    work: Callable[..., Any],
    items: Sequence[Any],
    workers: int,
    desc: str,
    unit: str = "video",
) -> Iterator[tuple[Any, Any]]:
    """Yield `(item, result)` as each item finishes, in completion order.

    Results arrive out of order, which is why the item is yielded alongside its
    result: callers key their bookkeeping off it rather than off position. A
    worker that raises propagates here, failing the run rather than silently
    dropping a video from the manifest.

    `workers <= 1` runs inline, keeping stack traces readable when debugging a
    single video and avoiding pool startup for a trial slice.
    """
    if not items:
        return

    if workers <= 1:
        bar = tqdm(items, desc=desc, unit=unit)
        for item in bar:
            bar.set_postfix_str(_label(item), refresh=False)
            yield item, work(item)
        return

    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = {pool.submit(work, item): item for item in items}
        bar = tqdm(total=len(futures), desc=desc, unit=unit)
        try:
            for future in as_completed(futures):
                item = futures[future]
                bar.set_postfix_str(_label(item), refresh=False)
                bar.update(1)
                yield item, future.result()
        finally:
            bar.close()


def _label(item: Any) -> str:
    return str(getattr(item, "video_id", None) or getattr(item, "stem", item))
