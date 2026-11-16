"""Importing shardguard caps the CUDA memory of PyTorch at the share of a GPU that the GPU shards of the Slurm job stand for.

The cap is the shards of the job over the shards one GPU of the node holds. shardguard puts it in the allocator configuration of PyTorch as ``per_process_memory_fraction``, which PyTorch reads when it initialises CUDA and which the processes the program starts inherit. If PyTorch has initialised CUDA already, it ignores the configuration, so shardguard also sets the cap with ``torch.cuda.set_per_process_memory_fraction``. The ``cudaMallocAsync`` backend ignores the configuration and can take more than a cap set otherwise, so shardguard refuses it. Outside a job holding shards, it does nothing.
"""

from __future__ import annotations

import logging
import math
import os
import re
import subprocess
import sys
from importlib import metadata
from typing import MutableMapping

logger = logging.getLogger(__name__)

SHARDS_PER_GPU = "SHARDGUARD_SHARDS_PER_GPU"
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
            "shardguard: the shards of this job are on several GPUs, which one fraction per GPU cannot cap"
        )
    per_gpu = environ.get(SHARDS_PER_GPU)
    if not per_gpu:
        try:
            per_gpu = shards_per_gpu(node_gres(environ["SLURMD_NODENAME"]))
        except (KeyError, OSError, subprocess.SubprocessError):
            per_gpu = None
        if not per_gpu:
            raise RuntimeError(
                f"shardguard: cannot tell how many shards one GPU of this node holds, so set {SHARDS_PER_GPU}"
            )
        environ[SHARDS_PER_GPU] = f"{per_gpu:g}"
    return min(1.0, int(shards) / float(per_gpu))


def _rounded(share: float) -> float:
    """`share` rounded down to four decimals."""
    return math.floor(share * 1e4) / 1e4


def with_cap(conf: str, share: float) -> str:
    """An allocator configuration capped at `share`. A lower cap it already has is kept."""
    settings = [item for item in conf.split(",") if item]
    capped = f"per_process_memory_fraction:{_rounded(share)}"
    match = FRACTION.search(conf)
    if match is None:
        settings.append(capped)
    elif float(match.group(1)) > share:
        settings = [capped if FRACTION.fullmatch(item) else item for item in settings]
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
            "shardguard: the cap needs PyTorch 2.10 or later, whose allocator configuration takes per_process_memory_fraction"
        )
    torch = sys.modules.get("torch")
    current = next((environ[name] for name in ALLOC_CONF if environ.get(name)), "")
    # PyTorch picks its allocator backend when it is imported.
    backend = current if torch is None else torch.cuda.get_allocator_backend()
    if "cudaMallocAsync" in backend:
        raise RuntimeError(
            "shardguard: the cudaMallocAsync backend can take more GPU memory than its cap"
        )
    capped = with_cap(current, share)
    for name in ALLOC_CONF:
        environ[name] = capped
    if torch is not None and torch.cuda.is_initialized():
        for device in range(torch.cuda.device_count()):
            if torch.cuda.get_per_process_memory_fraction(device) > share:
                torch.cuda.set_per_process_memory_fraction(_rounded(share), device)
    logger.info(
        "CUDA memory capped at %s of the GPU (%s of %s shards)",
        _rounded(share),
        environ["SLURM_SHARDS_ON_NODE"],
        environ[SHARDS_PER_GPU],
    )
    return share


cap()
