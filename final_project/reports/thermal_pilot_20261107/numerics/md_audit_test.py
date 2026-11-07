"""Protect the distinction between occupation retention and identical-Li exchange."""
import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("md_audit", Path(__file__).with_name("md_audit.py"))
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_same_occupation_can_contain_atom_resolved_assignment_changes(tmp_path):
    path = tmp_path / "mapping.jsonl"
    rows = []
    for assignment in [[0, 1], [1, 0], [1, 0]]:
        rows.append(dict(vacancy_mask_hungarian="0011", assigned_parent_li_sites=assignment,
                         changed_occupancy_sites=0, nearest_second_gap_A=[1., 1.],
                         assignment_distances_A=[.1, .1], hungarian_differs_from_nearest=0))
    path.write_text("".join(json.dumps(x) + "\n" for x in rows))
    result = audit.mapping_audit(path, "synthetic", "production", [0, 1])
    assert result["distinct_assigned_masks"] == 1
    assert result["fraction_changed_masks"] == 0
    assert result["unique_atom_resolved_assignments"] == 2
    assert result["Li_atoms_with_assignment_different_from_original"] == 2
    assert result["atom_resolved_assignment_transition_count"] == 2
    assert result["fraction_frames_atom_assignment_differs_from_original"] == 2 / 3
