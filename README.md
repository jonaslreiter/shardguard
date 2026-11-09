# safe-slurm

safe-slurm keeps a Slurm job within the GPU shards and the memory it was allocated. Install it with

```bash
pip install safe-slurm
```

and import it in the entry point of your program, before anything initialises CUDA:

```python
import safe_slurm  # both
import safe_shards  # GPU shards only
import safe_memory  # memory only
```

Outside a Slurm job, nothing happens.

## safe_shards

Slurm does not limit the GPU memory of a job that holds GPU shards, so the job can take memory that its co-tenants on the card were promised. `safe_shards` caps the CUDA memory of PyTorch at the share of a GPU that the shards of the job stand for. That share is the number of shards of the job over the number of shards one GPU of the node holds, which is read from `scontrol show node` or from `SAFE_SLURM_SHARDS_PER_GPU`. Past the cap, PyTorch raises an out-of-memory error.

The cap is `per_process_memory_fraction` in `PYTORCH_ALLOC_CONF` and `PYTORCH_CUDA_ALLOC_CONF`, together with `expandable_segments`. It needs PyTorch 2.10 or later with its native allocator, and it must be set before CUDA is initialised, so `safe_shards` raises an error otherwise. The cap applies to each process separately and does not cover the CUDA context, which takes a few hundred MB. It also assumes that the shards of the job are on one GPU. A job that holds whole GPUs is left alone.

## safe_memory

Slurm limits the RAM of a job to its allocation, but on a cluster that does not limit swap, a job past its allocation is swapped out without bound. With zram, that swap takes RAM from the whole node, and with disk swap, it takes swap space from every job on the node.

`safe_memory` starts a watcher process that reads the memory cgroup of the job every second, under cgroup v1 or v2. When the RAM and swap of the job, less the page cache the kernel could drop, exceed the allocation, the watcher kills every process of the job on the node, which ends the job. On a cluster where Slurm limits swap itself, `safe_memory` does nothing.
