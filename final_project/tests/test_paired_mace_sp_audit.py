"""An energy-zero change must leave ranks and centered errors unchanged."""
import importlib.util
from pathlib import Path
import sys

import numpy as np


def test_constant_teacher_shift_has_no_centered_error():
    scripts = Path(__file__).resolve().parents[1]/"scripts"/"overnight_20261108"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("mace_sp_audit_test", scripts/"paired_mace_sp_audit.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        a = module.summarize_energy(np.arange(4.), np.arange(4.)+2.5)
        assert a["spearman"] == a["kendall_tau_b"] == 1
        assert a["offset_removed_RMSE_meV_atom"] == 0
        assert a["mean_MACE_minus_UMA_offset_meV_atom"] == 2500
        assert a["pairwise_concordant"] == 6
        b = module.summarize_energy(np.arange(4.), -np.arange(4.)+2.5)
        assert b["spearman"] == b["kendall_tau_b"] == -1
        assert b["pairwise_discordant"] == 6
    finally:
        sys.path.remove(str(scripts))
