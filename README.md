# shardguard

Slurm can split a GPU into shards, but it does not limit the GPU memory of a job that holds shards, so the job can take memory that its co-tenants on the card were promised. shardguard caps the CUDA memory of PyTorch at the share of the GPU that the shards of the job stand for. Install it with

```bash
pip install shardguard
```

and import it in the entry point of your program, before it puts anything on the GPU:

```python
import shardguard
```

Past the cap, PyTorch raises an out-of-memory error. Outside a Slurm job that holds shards, nothing happens. shardguard logs the cap to the `shardguard` logger at level INFO, which shows only if logging is set up before shardguard is imported.

## How it works

The share is the number of shards of the job over the number of shards one GPU of the node holds, which shardguard reads from `scontrol show node`. Setting `SHARDGUARD_SHARDS_PER_GPU` gives that number instead. The cap goes in `PYTORCH_ALLOC_CONF` and `PYTORCH_CUDA_ALLOC_CONF` as `per_process_memory_fraction`, which PyTorch reads when it initialises CUDA and which the processes the program starts inherit. If PyTorch has initialised CUDA already, it ignores them, so shardguard also sets the cap with `torch.cuda.set_per_process_memory_fraction`. The cap counts the memory PyTorch reserves, not only what it allocates, so fragmentation eats into it. Setting `expandable_segments:True` in the same variables reduces that.

The cap needs PyTorch 2.10 or later with its native allocator. shardguard raises an error for the `cudaMallocAsync` backend, which ignores the cap in the allocator configuration and, capped otherwise, does not count the memory it keeps for reuse. The cap applies to each process separately and does not cover the CUDA context, which takes a few hundred MB. shardguard also raises an error when the shards of the job are spread over several GPUs.
