"""Importing this module ends the Slurm job once the memory it cannot give back exceeds its allocation.

Slurm holds the RAM of a job to its allocation, but where it does not limit swap, a job past its allocation is swapped out without bound. A watcher process reads the memory cgroup of the job every second. Once RAM and swap together, less the page cache the kernel could drop, exceed the allocation, it kills every process of the job on this node, which ends the job. Outside a Slurm job, where the kernel already limits the swap of the job, or without a readable memory cgroup with a limit, it does nothing.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

INTERVAL_SECONDS = 1.0
CGROUP_ROOT = Path("/sys/fs/cgroup")
# Set in the environment once a process of the job watches it, so that its child processes do not start another watcher.
WATCHED = "SAFE_SLURM_WATCHED"
# cgroup v1 writes no limit as 9223372036854771712, and v2 as "max".
UNLIMITED = 1 << 62

# Per cgroup version: the files that add up to RAM and swap, the keys of memory.stat to subtract (the page cache, and in v2 also the swap cache, which both files count), the RAM limit and the swap limit.
FILES = {
    "v1": (
        ("memory.memsw.usage_in_bytes",),
        ("total_active_file", "total_inactive_file"),
        "memory.limit_in_bytes",
        "memory.memsw.limit_in_bytes",
    ),
    "v2": (
        ("memory.current", "memory.swap.current"),
        ("active_file", "inactive_file", "swapcached"),
        "memory.max",
        "memory.swap.max",
    ),
}


def _limit(text: str) -> int | None:
    text = text.strip()
    return None if text == "max" or int(text) >= UNLIMITED else int(text)


def job_cgroup(
    job_id: str, proc_cgroup: str, root: Path = CGROUP_ROOT
) -> tuple[str, Path] | None:
    """The cgroup version and the memory cgroup of the job, given the contents of ``/proc/self/cgroup``, or None."""
    job = f"job_{job_id}"
    for line in proc_cgroup.splitlines():
        _, controllers, path = line.split(":", 2)
        parts = Path(path).parts
        if job not in parts:
            continue
        relative = parts[1 : parts.index(job) + 1]
        if "memory" in controllers.split(","):
            return "v1", root.joinpath("memory", *relative)
        if not controllers:
            return "v2", root.joinpath(*relative)
    return None


def used(version: str, directory: Path) -> int:
    """RAM and swap in use, less what the kernel could reclaim without swapping, in bytes."""
    usage, reclaimable, _, _ = FILES[version]
    stat = dict(
        line.split() for line in (directory / "memory.stat").read_text().splitlines()
    )
    total = sum(int((directory / name).read_text()) for name in usage)
    return total - sum(int(stat.get(key, 0)) for key in reclaimable)


def allocation(version: str, directory: Path) -> int | None:
    """The RAM limit, or None if there is none or the kernel limits the swap of the job itself."""
    _, _, ram, swap = FILES[version]
    if _limit((directory / swap).read_text()) is not None:
        return None
    return _limit((directory / ram).read_text())


def kill_job(directory: Path) -> None:
    """SIGKILL every process of the job on this node, this one last."""
    pids = {
        int(pid)
        for d, _, _ in os.walk(directory)
        for pid in (Path(d) / "cgroup.procs").read_text().split()
    }
    for pid in sorted(pids - {os.getpid()}) + [os.getpid()]:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def watch_memory(
    version: str,
    directory: Path,
    limit: int,
    parent: int,
    interval: float = INTERVAL_SECONDS,
    kill=kill_job,
) -> None:
    """While `parent` lives, compare the memory of the job with `limit` every `interval` seconds, and `kill` the job once it is over."""
    while os.getppid() == parent:
        time.sleep(interval)
        try:
            in_use = used(version, directory)
        except (OSError, ValueError):
            return
        if in_use > limit:
            print(
                f"safe_memory: the job holds {in_use >> 20} MiB of RAM and swap beyond its page cache, over its allocation of {limit >> 20} MiB, so it is ended",
                file=sys.stderr,
                flush=True,
            )
            kill(directory)
            return


def watch(environ=os.environ) -> subprocess.Popen | None:
    """Start the watcher process and return it, or None if there is nothing to watch."""
    job_id = environ.get("SLURM_JOB_ID")
    if not job_id or environ.get(WATCHED) == job_id:
        return None
    cgroup = job_cgroup(job_id, Path("/proc/self/cgroup").read_text())
    try:
        limit = allocation(*cgroup) if cgroup else None
        if limit:
            used(*cgroup)
    except (OSError, ValueError):
        limit = None
    if not limit:
        print(
            "safe_memory: memory not watched, since the job has no readable memory cgroup with a RAM limit and unlimited swap",
            file=sys.stderr,
        )
        return None
    version, directory = cgroup
    watcher = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-S",
            __file__,
            version,
            str(directory),
            str(limit),
            str(os.getpid()),
        ],
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    environ[WATCHED] = job_id
    print(f"safe_memory: memory held to {limit >> 20} MiB", file=sys.stderr)
    return watcher


if __name__ == "__main__":
    version, directory, limit, parent = sys.argv[1:]
    watch_memory(version, Path(directory), int(limit), int(parent))
else:
    watcher = watch()
