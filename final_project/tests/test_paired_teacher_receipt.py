"""Actual registry path shape versus executed calculator basename receipts."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/overnight_20261108"))
import paired_common as common
from cathode_cvae.mlip.registry import MODELS


def test_actual_registry_relative_hash_paths_match_executed_basenames_strictly():
    spec = MODELS["uma"]
    expected = dict(head=spec["head"], package_version=spec["version"], dtype="float32", tf32=False,
                    checkpoint_hashes=spec["files"])
    context = dict(teacher=expected, optimizer=dict(ase_version="3.26.0"))
    executed = dict(expected, checkpoint_hashes={Path(k).name:v for k,v in spec["files"].items()}, ase="3.26.0")
    common.verify_teacher_receipt(executed, context)
    bad = dict(executed, checkpoint_hashes={next(iter(executed["checkpoint_hashes"])):"0" * 64})
    with pytest.raises(ValueError, match="SHA256"):
        common.verify_teacher_receipt(bad, context)
    bad = dict(executed, tf32=True)
    with pytest.raises(ValueError, match="tf32"):
        common.verify_teacher_receipt(bad, context)


def test_duplicate_basename_or_truncated_hash_never_normalizes_away():
    with pytest.raises(ValueError, match="duplicate"):
        common.normalized_checkpoint_hashes({"first/model.pt":"a" * 64, "second/model.pt":"a" * 64})
    with pytest.raises(ValueError, match="complete SHA256"):
        common.normalized_checkpoint_hashes({"model.pt":"abc"})
