"""
Shared resume/checkpoint helper for run_e1.py and run_e3.py.

arq.experiments' drivers (e1_calibrate, e3_curve, e5_stress) have no
mid-run checkpointing by design ("nothing aggregated in memory across runs,
so a crash costs one file" -- code/arq/experiments.py's own docstring).
That's fine at a handful of pools, but once CoPHEME's pool count grew from
3 to 413 (10 Sep 2026, after the montage-generation jobs completed), a
single E1/E3 run means order-10^5-10^6 model calls -- long enough that a
SLURM walltime kill mid-run losing everything is a real risk.

CachingAgent (code/arq/agent.py) already persists an on-disk cache keyed by
(model, group, claim), but only flushes when .flush() is called explicitly
-- which run_e1.py/run_e3.py previously only did once, after the whole
driver call returned. PeriodicFlushAgent wraps that cache to flush every
FLUSH_EVERY calls, on process exit, and on SIGTERM (SLURM's walltime-kill
signal), so a kill loses at most one flush interval's work. Re-running the
same script afterward reloads the cache from disk; every previously-seen
(pool, group, claim) is a cache hit, not a new model call, so a resumed run
is fast until it reaches genuinely new ground. The underlying calibration
loop itself still restarts from pool 0 -- this is NOT true mid-loop
resume, it is "redo the loop, but skip anything already cached" -- which is
sufficient here because the loop order is deterministic (same seed, same
pool order) and the cache key doesn't depend on loop position.

This is an orchestration-level fix living in scripts/, not a change to
arq's core package -- e1_calibrate/e3_curve/e5_stress are unmodified.
"""
from __future__ import annotations
import atexit
import signal
import sys

from arq.agent import Agent, CachingAgent  # noqa: E402

FLUSH_EVERY = 200  # calls between cache flushes; tune down if a single
                    # flush interval's lost work becomes a concern at this
                    # scale (each flush rewrites the whole cache file, so
                    # too-frequent flushing has its own I/O cost at ~10^6
                    # cached entries).


class PeriodicFlushAgent(Agent):
    """Wraps a CachingAgent to flush to disk every flush_every calls, plus
    on process exit and on SIGTERM (SLURM's walltime-kill signal)."""

    def __init__(self, inner: CachingAgent, flush_every: int = FLUSH_EVERY):
        super().__init__(name=inner.name)
        self.inner, self.flush_every = inner, flush_every
        self._since_flush = 0
        atexit.register(self._safe_flush)
        signal.signal(signal.SIGTERM, self._on_sigterm)

    def is_mock(self) -> bool:
        return self.inner.is_mock()

    def _safe_flush(self):
        try:
            self.inner.flush()
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: cache flush failed: {e}", file=sys.stderr)

    def _on_sigterm(self, signum, frame):
        print(f"\nSIGTERM received ({self.inner.misses} new calls this run) "
              "-- flushing cache before exit.", file=sys.stderr)
        self._safe_flush()
        sys.exit(143)  # 128 + SIGTERM(15), conventional exit code

    def verdict(self, items, claim):
        v = self.inner.verdict(items, claim)
        self._since_flush += 1
        if self._since_flush >= self.flush_every:
            self._safe_flush()
            self._since_flush = 0
        return v


def wrap_with_resume(inner_agent: Agent, cache_path: str) -> Agent:
    """Convenience: CachingAgent(inner_agent, cache_path) wrapped in a
    PeriodicFlushAgent. This is what run_e1.py/run_e3.py should use instead
    of calling CachingAgent directly when --cache is set."""
    return PeriodicFlushAgent(CachingAgent(inner_agent, cache_path))


def final_flush(agent: Agent) -> None:
    """Call at the end of a driver run regardless of which wrapper (if any)
    was used, so nothing relies on atexit alone having fired correctly."""
    if hasattr(agent, "_safe_flush"):
        agent._safe_flush()  # PeriodicFlushAgent
    elif hasattr(agent, "flush"):
        agent.flush()  # bare CachingAgent
