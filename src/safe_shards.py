"""Importing this module caps the CUDA memory of PyTorch at the share of a GPU that the GPU shards of the Slurm job stand for.

The cap is ``per_process_memory_fraction`` in the allocator configuration of PyTorch, set to the shards of the job over the shards one GPU of the node holds. ``expandable_segments`` keeps what PyTorch reserves close to what it allocates. PyTorch reads the configuration when it initialises CUDA, so this module must be imported before that, which it checks. Outside a job holding shards, it does nothing.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import sys
from importlib import metadata
from typing import MutableMapping

SHARDS_PER_GPU = "SAFE_SLURM_SHARDS_PER_GPU"
# PyTorch reads the first of these that is set, and releases before 2.9 only the second.
ALLOC_CONF = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")
FRACTION = re.compile(r"per_process_memory_fraction:([0-9.]+)")
# The first release whose allocator configuration takes per_process_memory_fraction.
OLDEST_TORCH = (2, 10)


def shards_per_gpu(gres: str) -> float | None:
    """Shards per GPU from the ``Gres`` of a node, e.g. 24 for ``gpu:A10:4,shard:A10:96``, or None if it lists no GPUs or no shards."""
    counts = {"gpu": 0, "shard": 0}
    for entry in re.sub(r"\([^)]*\)", "", gres).split(","):
        fields = entry.split(":")
        if fields[0] in counts and fields[-1].isdigit():
            counts[fields[0]] += int(fields[-1])
    if not counts["gpu"] or not counts["shard"]:
        return None
    return counts["shard"] / counts["gpu"]


def node_gres(node: str) -> str:
    """The ``Gres`` that ``scontrol show node`` reports for `node`."""
    output = subprocess.run(
        ["scontrol", "show", "node", node],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    ).stdout
    match = re.search(r"\bGres=(\S*)", output)
    return match.group(1) if match else ""


def fraction(environ: MutableMapping[str, str] = os.environ) -> float | None:
    """The share of one GPU that the shards of the job stand for, or None outside a job holding shards."""
    shards = environ.get("SLURM_SHARDS_ON_NODE")
    if not shards:
        return None
    if "," in environ.get("CUDA_VISIBLE_DEVICES", ""):
        raise RuntimeError(
            "safe_shards: the shards of this job are on several GPUs, which one fraction per GPU cannot cap"
        )
    per_gpu = environ.get(SHARDS_PER_GPU)
    if not per_gpu:
        try:
            per_gpu = shards_per_gpu(node_gres(environ["SLURMD_NODENAME"]))
        except (KeyError, OSError, subprocess.SubprocessError):
            per_gpu = None
        if not per_gpu:
            raise RuntimeError(
                f"safe_shards: cannot tell how many shards one GPU of this node holds, so set {SHARDS_PER_GPU}"
            )
        environ[SHARDS_PER_GPU] = f"{per_gpu:g}"
    return min(1.0, int(shards) / float(per_gpu))


def _rounded(share: float) -> float:
    """`share` rounded down to four decimals."""
    return math.floor(share * 1e4) / 1e4


def with_cap(conf: str, share: float) -> str:
    """An allocator configuration capped at `share`, with expandable segments unless it sets them itself. A lower cap it already has is kept."""
    settings = [item for item in conf.split(",") if item]
    capped = f"per_process_memory_fraction:{_rounded(share)}"
    match = FRACTION.search(conf)
    if match is None:
        settings.append(capped)
    elif float(match.group(1)) > share:
        settings = [capped if FRACTION.fullmatch(item) else item for item in settings]
    if not any(item.startswith("expandable_segments:") for item in settings):
        settings.append("expandable_segments:True")
    return ",".join(settings)


def torch_version() -> tuple[int, int] | None:
    """The major and minor version of the installed PyTorch, or None if it is not installed."""
    try:
        version = metadata.version("torch")
    except metadata.PackageNotFoundError:
        return None
    major, minor = re.match(r"(\d+)\.(\d+)", version).groups()
    return int(major), int(minor)


def cap(environ: MutableMapping[str, str] = os.environ) -> float | None:
    """Cap the CUDA memory of PyTorch at the shards of the job and return the share, or None outside a job holding shards."""
    share = fraction(environ)
    if share is None:
        return None
    version = torch_version()
    if version is None or version < OLDEST_TORCH:
        raise RuntimeError(
            "safe_shards: the cap needs PyTorch 2.10 or later, whose allocator configuration takes per_process_memory_fraction"
        )
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        raise RuntimeError(
            "safe_shards: import it before CUDA is initialised, since PyTorch ignores the cap after that"
        )
    current = next((environ[name] for name in ALLOC_CONF if environ.get(name)), "")
    if "cudaMallocAsync" in current:
        raise RuntimeError(
            "safe_shards: the cudaMallocAsync backend ignores per_process_memory_fraction"
        )
    capped = with_cap(current, share)
    for name in ALLOC_CONF:
        environ[name] = capped
    print(
        f"safe_shards: CUDA memory capped at {_rounded(share)} of a GPU, the share of {environ['SLURM_SHARDS_ON_NODE']} of its {environ[SHARDS_PER_GPU]} shards",
        file=sys.stderr,
    )
    return share


cap()
