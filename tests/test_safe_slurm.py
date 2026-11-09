import safe_memory
import safe_shards
import safe_slurm


def test_the_package_turns_on_both_modules():
    assert safe_slurm.safe_memory is safe_memory
    assert safe_slurm.safe_shards is safe_shards
