import sys
import types

import pytest

import shardguard


@pytest.fixture(autouse=True)
def torch_2_14(monkeypatch):
    monkeypatch.setattr(shardguard, "torch_version", lambda: (2, 14))


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
    assert shardguard.shards_per_gpu(gres) == expected


def test_outside_a_shard_job_nothing_is_capped():
    environ = {"SLURM_JOB_ID": "1"}
    assert shardguard.cap(environ) is None
    assert environ == {"SLURM_JOB_ID": "1"}


def test_the_cap_is_the_jobs_share_of_a_gpu():
    environ = {"SLURM_SHARDS_ON_NODE": "6", shardguard.SHARDS_PER_GPU: "24"}
    assert shardguard.cap(environ) == 0.25
    for name in shardguard.ALLOC_CONF:
        assert environ[name] == "per_process_memory_fraction:0.25"


def test_the_shards_per_gpu_come_from_the_node_once(monkeypatch):
    calls = []
    monkeypatch.setattr(
        shardguard,
        "node_gres",
        lambda node: calls.append(node) or "gpu:A10:4,shard:A10:96",
    )
    environ = {"SLURM_SHARDS_ON_NODE": "12", "SLURMD_NODENAME": "cn-1"}
    assert shardguard.fraction(environ) == 0.5
    assert shardguard.fraction(environ) == 0.5
    assert calls == ["cn-1"]


@pytest.mark.parametrize("gres", ["gpu:A10:4", OSError, KeyError])
def test_a_node_that_cannot_say_its_shards_is_refused(monkeypatch, gres):
    def node_gres(node):
        if isinstance(gres, str):
            return gres
        raise gres

    monkeypatch.setattr(shardguard, "node_gres", node_gres)
    with pytest.raises(RuntimeError, match=shardguard.SHARDS_PER_GPU):
        shardguard.fraction({"SLURM_SHARDS_ON_NODE": "4", "SLURMD_NODENAME": "cn-1"})


def test_shards_on_several_gpus_are_refused():
    with pytest.raises(RuntimeError, match="several GPUs"):
        shardguard.fraction(
            {"SLURM_SHARDS_ON_NODE": "4", "CUDA_VISIBLE_DEVICES": "0,1"}
        )


def settings(conf):
    return dict(item.split(":", 1) for item in conf.split(","))


def test_existing_settings_are_kept_and_a_lower_cap_wins():
    assert settings(shardguard.with_cap("expandable_segments:False", 0.5)) == {
        "expandable_segments": "False",
        "per_process_memory_fraction": "0.5",
    }
    assert settings(shardguard.with_cap("per_process_memory_fraction:0.1", 0.5)) == {
        "per_process_memory_fraction": "0.1",
    }
    assert (
        settings(shardguard.with_cap("per_process_memory_fraction:0.9", 0.5))[
            "per_process_memory_fraction"
        ]
        == "0.5"
    )


def test_the_cap_is_rounded_down_and_other_settings_are_left_alone():
    conf = shardguard.with_cap("roundup_power2_divisions:[256:1,512:2]", 1 / 6)
    assert (
        conf
        == "roundup_power2_divisions:[256:1,512:2],per_process_memory_fraction:0.1666"
    )


def test_old_or_missing_pytorch_is_refused(monkeypatch):
    for version in ((2, 9), None):
        monkeypatch.setattr(shardguard, "torch_version", lambda: version)
        with pytest.raises(RuntimeError, match="2.10"):
            shardguard.cap(
                {"SLURM_SHARDS_ON_NODE": "4", shardguard.SHARDS_PER_GPU: "24"}
            )


def test_before_pytorch_is_imported_the_async_allocator_is_refused():
    with pytest.raises(RuntimeError, match="cudaMallocAsync"):
        shardguard.cap(
            {
                "SLURM_SHARDS_ON_NODE": "4",
                shardguard.SHARDS_PER_GPU: "24",
                "PYTORCH_ALLOC_CONF": "backend:cudaMallocAsync",
            }
        )


class Cuda:
    """The memory fractions of ``torch.cuda`` for `devices` GPUs, each at `fraction`."""

    def __init__(self, devices, fraction=1.0):
        self.fractions = [fraction] * devices

    def device_count(self):
        return len(self.fractions)

    def get_per_process_memory_fraction(self, device):
        return self.fractions[device]

    def set_per_process_memory_fraction(self, fraction, device):
        self.fractions[device] = fraction


def import_torch(monkeypatch, devices=1, fraction=1.0):
    cuda = Cuda(devices, fraction)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
    return cuda


def test_after_pytorch_is_imported_each_gpu_is_capped_directly_too(monkeypatch):
    cuda = import_torch(monkeypatch, devices=2)
    environ = {"SLURM_SHARDS_ON_NODE": "4", shardguard.SHARDS_PER_GPU: "24"}
    assert shardguard.cap(environ) == 4 / 24
    assert cuda.fractions == [0.1666, 0.1666]
    for name in shardguard.ALLOC_CONF:
        assert environ[name] == "per_process_memory_fraction:0.1666"


def test_after_pytorch_is_imported_a_lower_cap_wins(monkeypatch):
    cuda = import_torch(monkeypatch, fraction=0.1)
    shardguard.cap({"SLURM_SHARDS_ON_NODE": "12", shardguard.SHARDS_PER_GPU: "24"})
    assert cuda.fractions == [0.1]


def test_after_pytorch_is_imported_the_async_allocator_is_capped(monkeypatch):
    cuda = import_torch(monkeypatch)
    shardguard.cap(
        {
            "SLURM_SHARDS_ON_NODE": "6",
            shardguard.SHARDS_PER_GPU: "24",
            "PYTORCH_ALLOC_CONF": "backend:cudaMallocAsync",
        }
    )
    assert cuda.fractions == [0.25]
