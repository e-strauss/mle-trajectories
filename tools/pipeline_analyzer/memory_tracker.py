"""Reusable RSS memory tracker that runs in a separate process.

Usage::

    from memory_tracker import MemoryTracker

    tracker = MemoryTracker(dump_path="mem.csv")
    tracker.start()
    try:
        ... # workload
    finally:
        samples = tracker.stop()   # list of (wall_sec, rss_mb)
        tracker.write_csv("memory_usage.csv", t0=start_time)
"""

from __future__ import annotations

import os
import signal
from multiprocessing import get_context
from time import perf_counter
from typing import Literal

import psutil


def _flush_samples(samples_list, dump_path: str | None, flushed_count: int) -> int:
    if not dump_path:
        return flushed_count
    new_samples = list(samples_list[flushed_count:])
    if not new_samples:
        return flushed_count
    mode = "a" if flushed_count > 0 else "w"
    with open(dump_path, mode) as f:
        if flushed_count == 0:
            f.write("time_sec,rss_mb\n")
        for ts, rss_mb in new_samples:
            f.write(f"{ts:.6f},{rss_mb:.2f}\n")
    return flushed_count + len(new_samples)


def _tracker_loop(
    pid: int,
    stop_event,
    samples_list,
    interval_sec: float,
    dump_path: str | None,
    flush_every: int,
) -> None:
    flushed = 0

    def _sigterm_handler(signum, frame):
        nonlocal flushed
        flushed = _flush_samples(samples_list, dump_path, flushed)
        raise SystemExit(1)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    sample_count = 0

    def _sample(t, rss_mb):
        nonlocal sample_count, flushed
        samples_list.append((t, rss_mb))
        sample_count += 1
        if sample_count % flush_every == 0:
            flushed = _flush_samples(samples_list, dump_path, flushed)

    if pid != -1:
        try:
            proc = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return
        while not stop_event.is_set():
            try:
                rss_bytes = proc.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            _sample(perf_counter(), rss_bytes / (1024 * 1024))
            stop_event.wait(interval_sec)
    else:
        parent = psutil.Process(os.getppid())
        while not stop_event.is_set():
            if not parent.is_running():
                break
            _sample(perf_counter(), psutil.virtual_memory().used / (1024 * 1024))
            stop_event.wait(interval_sec)

    _flush_samples(samples_list, dump_path, flushed)


class MemoryTracker:
    """Spawn a side-car process that polls RSS of the current (or system) memory.

    Parameters
    ----------
    mode : "process" or "system"
        "process" tracks the calling process's RSS.
        "system" tracks total system memory used (useful with multi-process workloads).
    interval_sec : float
        Polling interval in seconds.
    live_dump_path : str or None
        If set, samples are incrementally flushed to this CSV while tracking.
        ``None`` keeps the samples in memory only.
    flush_every : int
        Number of samples between live flushes.

    Notes
    -----
    The side-car is started with the ``fork`` start method, not the interpreter
    default. Python 3.14 defaults to ``forkserver`` on Linux, which starts the
    tracker as a *fresh* interpreter that has to re-import the module holding
    ``_tracker_loop`` -- inside a workload process that has already imported
    pandas/sklearn/torch that costs seconds, and the tracker misses exactly the
    ramp-up it was started to observe (measured: first sample 2.1 s late, against
    ~10 ms with fork). The side-car only reads ``/proc`` and appends to a file,
    so it needs nothing from a clean interpreter.
    """

    def __init__(
        self,
        *,
        mode: Literal["process", "system"] = "process",
        interval_sec: float = 0.1,
        live_dump_path: str | None = "memory_usage_live.csv",
        flush_every: int = 50,
    ):
        self.mode = mode
        self.interval_sec = interval_sec
        self.live_dump_path = live_dump_path
        self.flush_every = flush_every

        self._ctx = get_context("fork")
        self._manager = self._ctx.Manager()
        self._samples = self._manager.list()
        self._stop = self._ctx.Event()
        self._process = None
        self._t0: float | None = None

    def start(self) -> None:
        pid = os.getpid() if self.mode == "process" else -1
        self._t0 = perf_counter()
        self._process = self._ctx.Process(
            target=_tracker_loop,
            args=(pid, self._stop, self._samples, self.interval_sec,
                  self.live_dump_path, self.flush_every),
        )
        self._process.start()

    def stop(self, timeout: float = 2.0) -> list[tuple[float, float]]:
        """Signal the tracker to stop and return collected samples."""
        self._stop.set()
        if self._process is not None:
            self._process.join(timeout=timeout)
        return list(self._samples)

    @property
    def t0(self) -> float:
        if self._t0 is None:
            raise RuntimeError("Tracker has not been started yet")
        return self._t0

    def write_csv(self, path: str, *, t0: float | None = None) -> None:
        """Write all samples to *path* with wall-clock times relative to *t0*."""
        t0 = t0 if t0 is not None else self._t0
        if t0 is None:
            raise RuntimeError("t0 not available; pass it explicitly or call start() first")
        samples = list(self._samples)
        if not samples:
            return
        with open(path, "w") as f:
            f.write("time_sec,rss_mb\n")
            for ts, rss_mb in samples:
                f.write(f"{ts - t0:.4f},{rss_mb:.2f}\n")
