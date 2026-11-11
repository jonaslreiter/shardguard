# shardguard

Slurm can split a GPU into shards, but it does not limit the GPU memory of a job that holds shards, so the job can take memory that its co-tenants on the card were promised. shardguard caps the CUDA memory of PyTorch at the share of the GPU that the shards of the job stand for. Install it with

```bash
pip install shardguard
```

and import it in the entry point of your program, before anything initialises CUDA:

```python
import shardguard
```

Past the cap, PyTorch raises an out-of-memory error. Outside a Slurm job that holds shards, nothing happens.

## How it works

The share is the number of shards of the job over the number of shards one GPU of the node holds, which shardguard reads from `scontrol show node`. Setting `SHARDGUARD_SHARDS_PER_GPU` gives that number instead. The cap is `per_process_memory_fraction` in `PYTORCH_ALLOC_CONF` and `PYTORCH_CUDA_ALLOC_CONF`, together with `expandable_segments`, which keeps the memory PyTorch reserves close to what it allocates.

The cap needs PyTorch 2.10 or later with its native allocator. PyTorch reads the setting when it initialises CUDA, so shardguard raises an error when it is imported after that. The cap applies to each process separately and does not cover the CUDA context, which takes a few hundred MB. shardguard also raises an error when the shards of the job are spread over several GPUs.
