"""Keep a Slurm job within the GPU shards and the memory it was allocated.

``import safe_slurm`` does both, and ``import safe_shards`` or ``import safe_memory`` does one. Import it in the entry point of the program, before anything initialises CUDA. Outside a Slurm job it does nothing.
"""

import safe_memory  # noqa: F401
import safe_shards  # noqa: F401
