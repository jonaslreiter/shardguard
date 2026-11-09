import os

# Importing the modules acts on a Slurm job, so the tests run as outside one.
for name in ("SLURM_JOB_ID", "SLURM_SHARDS_ON_NODE"):
    os.environ.pop(name, None)
