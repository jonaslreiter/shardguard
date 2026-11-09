import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import safe_memory

V1_UNLIMITED = 9223372036854771712


def cgroup(directory: Path, **files) -> Path:
    directory.mkdir(parents=True)
    for name, value in files.items():
        (directory / name.replace("__", ".")).write_text(f"{value}\n")
    return directory


def v1(tmp_path, usage=300, limit=1000, swap_limit=V1_UNLIMITED, cache=0):
    return cgroup(
        tmp_path / "memory/slurm/uid_1/job_7",
        memory__memsw__usage_in_bytes=usage,
        memory__limit_in_bytes=limit,
        memory__memsw__limit_in_bytes=swap_limit,
        memory__stat=f"cache {cache}\ntotal_active_file {cache // 2}\ntotal_inactive_file {cache - cache // 2}",
    )


def v2(
    tmp_path, current=300, swap=200, limit=1000, swap_limit="max", cache=0, swapcached=0
):
    return cgroup(
        tmp_path / "system.slice/slurmstepd.scope/job_7",
        memory__current=current,
        memory__swap__current=swap,
        memory__max=limit,
        memory__swap__max=swap_limit,
        memory__stat=f"active_file {cache}\ninactive_file 0\nswapcached {swapcached}",
    )


V1_LINE = "5:memory,hugetlb:/slurm/uid_1/job_7/step_batch/task_0\n"
V2_LINE = "0::/system.slice/slurmstepd.scope/job_7/step_batch/user/task_0\n"


def test_the_job_cgroup_is_found_for_both_versions(tmp_path):
    assert safe_memory.job_cgroup("7", V1_LINE, tmp_path) == (
        "v1",
        tmp_path / "memory/slurm/uid_1/job_7",
    )
    assert safe_memory.job_cgroup("7", V2_LINE, tmp_path) == (
        "v2",
        tmp_path / "system.slice/slurmstepd.scope/job_7",
    )
    assert (
        safe_memory.job_cgroup("7", "0::/user.slice/session-1.scope\n", tmp_path)
        is None
    )


def test_page_cache_does_not_count(tmp_path):
    assert safe_memory.used("v1", v1(tmp_path, usage=900, cache=600)) == 300
    assert (
        safe_memory.used(
            "v2", v2(tmp_path, current=900, swap=200, cache=600, swapcached=100)
        )
        == 400
    )


def test_the_allocation_is_the_ram_limit_where_swap_is_unlimited(tmp_path):
    assert safe_memory.allocation("v1", v1(tmp_path / "a")) == 1000
    assert safe_memory.allocation("v2", v2(tmp_path / "b")) == 1000


def test_nothing_is_watched_where_the_kernel_limits_swap_or_ram_is_unlimited(tmp_path):
    assert safe_memory.allocation("v1", v1(tmp_path / "a", swap_limit=1500)) is None
    assert safe_memory.allocation("v2", v2(tmp_path / "b", swap_limit=0)) is None
    assert safe_memory.allocation("v1", v1(tmp_path / "c", limit=V1_UNLIMITED)) is None
    assert safe_memory.allocation("v2", v2(tmp_path / "d", limit="max")) is None


def test_the_job_is_killed_once_over_its_allocation(monkeypatch):
    readings = iter([900, 1000, 1001, 2000])
    monkeypatch.setattr(safe_memory, "used", lambda version, directory: next(readings))
    killed = []
    safe_memory.watch_memory("v1", Path("job"), 1000, os.getppid(), 0, killed.append)
    assert killed == [Path("job")]
    assert next(readings) == 2000


def test_the_watch_ends_with_the_cgroup_or_its_parent(tmp_path, monkeypatch):
    safe_memory.watch_memory(
        "v2", tmp_path / "gone", 1000, os.getppid(), 0, pytest.fail
    )
    monkeypatch.setattr(safe_memory, "used", lambda version, directory: 2000)
    safe_memory.watch_memory("v2", tmp_path, 1000, os.getppid() + 1, 0, pytest.fail)


def test_killing_the_job_kills_every_process_in_its_cgroup(tmp_path, monkeypatch):
    job = tmp_path / "job_7"
    (job / "step_batch").mkdir(parents=True)
    (job / "cgroup.procs").write_text("")
    (job / "step_batch/cgroup.procs").write_text(f"11\n12\n{os.getpid()}\n")
    killed = []
    monkeypatch.setattr(os, "kill", lambda pid, signal: killed.append(pid))
    safe_memory.kill_job(job)
    assert killed == [11, 12, os.getpid()]


def test_outside_a_job_or_once_watched_nothing_starts():
    assert safe_memory.watch({}) is None
    assert safe_memory.watch({"SLURM_JOB_ID": "7", safe_memory.WATCHED: "7"}) is None


def test_the_watcher_runs_as_a_script_and_exits_with_its_parent(tmp_path):
    directory = v1(tmp_path)
    child = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            safe_memory.__file__,
            "v1",
            str(directory),
            "1000",
            "1",
        ],
        timeout=10,
    )
    assert child.returncode == 0
