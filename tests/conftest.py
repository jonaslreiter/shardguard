import os

# Importing shardguard acts on a Slurm job that holds GPU shards, so the tests run as outside one.
os.environ.pop("SLURM_SHARDS_ON_NODE", None)
