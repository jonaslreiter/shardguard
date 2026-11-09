import sys
import types

import pytest

import safe_shards


@pytest.fixture(autouse=True)
def torch_2_14(monkeypatch):
    monkeypatch.setattr(safe_shards, "torch_version", lambda: (2, 14))


@pytest.mark.parametrize(
    "gres, expected",
    [
        ("gpu:A10:4,shard:A10:96", 24),
        ("gpu:l40s:2(S:0-1),shard:l40s:96(S:0-1)", 48),
        ("gpu:4", None),
        ("(null)", None),
    ],
)
def test_shards_per_gpu_is_read_off_the_gres(gres, expected):
    assert safe_shards.shards_per_gpu(gres) == expected


def test_outside_a_shard_job_nothing_is_capped():
    environ = {"SLURM_JOB_ID": "1"}
    assert safe_shards.cap(environ) is None
    assert environ == {"SLURM_JOB_ID": "1"}


def test_the_cap_is_the_jobs_share_of_a_gpu():
    environ = {"SLURM_SHARDS_ON_NODE": "6", safe_shards.SHARDS_PER_GPU: "24"}
    assert safe_shards.cap(environ) == 0.25
    for name in safe_shards.ALLOC_CONF:
        assert (
            environ[name] == "per_process_memory_fraction:0.25,expandable_segments:True"
        )


def test_the_shards_per_gpu_come_from_the_node_once(monkeypatch):
    calls = []
    monkeypatch.setattr(
        safe_shards,
        "node_gres",
        lambda node: calls.append(node) or "gpu:A10:4,shard:A10:96",
    )
    environ = {"SLURM_SHARDS_ON_NODE": "12", "SLURMD_NODENAME": "cn-1"}
    assert safe_shards.fraction(environ) == 0.5
    assert safe_shards.fraction(environ) == 0.5
    assert calls == ["cn-1"]


@pytest.mark.parametrize("gres", ["gpu:A10:4", OSError, KeyError])
def test_a_node_that_cannot_say_its_shards_is_refused(monkeypatch, gres):
    def node_gres(node):
        if isinstance(gres, str):
            return gres
        raise gres

    monkeypatch.setattr(safe_shards, "node_gres", node_gres)
    with pytest.raises(RuntimeError, match=safe_shards.SHARDS_PER_GPU):
        safe_shards.fraction({"SLURM_SHARDS_ON_NODE": "4", "SLURMD_NODENAME": "cn-1"})


def test_shards_on_several_gpus_are_refused():
    with pytest.raises(RuntimeError, match="several GPUs"):
        safe_shards.fraction(
            {"SLURM_SHARDS_ON_NODE": "4", "CUDA_VISIBLE_DEVICES": "0,1"}
        )


def settings(conf):
    return dict(item.split(":", 1) for item in conf.split(","))


def test_existing_settings_are_kept_and_a_lower_cap_wins():
    assert settings(safe_shards.with_cap("expandable_segments:False", 0.5)) == {
        "expandable_segments": "False",
        "per_process_memory_fraction": "0.5",
    }
    assert settings(safe_shards.with_cap("per_process_memory_fraction:0.1", 0.5)) == {
        "expandable_segments": "True",
        "per_process_memory_fraction": "0.1",
    }
    assert (
        settings(safe_shards.with_cap("per_process_memory_fraction:0.9", 0.5))[
            "per_process_memory_fraction"
        ]
        == "0.5"
    )


def test_the_cap_is_rounded_down_and_other_settings_are_left_alone():
    conf = safe_shards.with_cap("roundup_power2_divisions:[256:1,512:2]", 1 / 6)
    assert (
        conf
        == "roundup_power2_divisions:[256:1,512:2],per_process_memory_fraction:0.1666,expandable_segments:True"
    )


def test_old_or_missing_pytorch_is_refused(monkeypatch):
    for version in ((2, 9), None):
        monkeypatch.setattr(safe_shards, "torch_version", lambda: version)
        with pytest.raises(RuntimeError, match="2.10"):
            safe_shards.cap(
                {"SLURM_SHARDS_ON_NODE": "4", safe_shards.SHARDS_PER_GPU: "24"}
            )


def test_the_async_allocator_is_refused():
    with pytest.raises(RuntimeError, match="cudaMallocAsync"):
        safe_shards.cap(
            {
                "SLURM_SHARDS_ON_NODE": "4",
                safe_shards.SHARDS_PER_GPU: "24",
                "PYTORCH_ALLOC_CONF": "backend:cudaMallocAsync",
            }
        )


def test_importing_after_cuda_is_initialised_is_refused(monkeypatch):
    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_initialized=lambda: True)
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    with pytest.raises(RuntimeError, match="before CUDA"):
        safe_shards.cap({"SLURM_SHARDS_ON_NODE": "4", safe_shards.SHARDS_PER_GPU: "24"})
