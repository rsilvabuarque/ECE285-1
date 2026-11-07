"""Describe completed-frame Li assignment events without imposing a hop cutoff."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from collections import deque

import numpy as np
from ase.geometry import find_mic


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(mask, permutations):
    v = np.array(list(mask), dtype=int)
    return min("".join(map(str, v[p])) for p in permutations)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--md-root", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    answers = []; context = []; sources = {}
    for receipt_path in sorted(args.md_root.glob("*/*/receipt.json")):
        receipt = json.loads(receipt_path.read_text()); assert receipt["status"] == "COMPLETE"
        identity = str(receipt_path.parent.relative_to(args.md_root))
        baseline = tuple(receipt["initial_mapping"]["assigned_parent_li_sites"])
        baseline_mask = receipt["initial_mapping"]["vacancy_mask_hungarian"]
        phases = {}
        sources[str(receipt_path)] = sha(receipt_path)
        with np.load(receipt["arguments"]["parent"]) as z:
            parent_li_indices = z["li_indices"]; parent_positions = z["positions_A"]; cell = z["cell_A"]; permutations = z["permutations"]
        with np.load(receipt_path.parent / "production.npz") as z:
            numbers = z["numbers"]; parent_indices = z["parent_indices"]
        li_rows = np.flatnonzero(numbers == 3)
        assert len(li_rows) == len(baseline)
        changed_any_phase = False
        for phase in ["equilibration", "production"]:
            path = receipt_path.parent / f"{phase}_mapping.jsonl"
            sources[str(path)] = sha(path)
            states = []; state = None; before = deque(maxlen=12); first = None; affected = set(); events_context = []
            n = 0; changed_frames = 0; previous = None; transitions = np.zeros(len(baseline), int)
            with path.open() as stream:
                for line in stream:
                    item = json.loads(line); n += 1
                    assignment = tuple(item["assigned_parent_li_sites"])
                    changed = [i for i, (old, new) in enumerate(zip(baseline, assignment)) if old != new]
                    if previous is not None: transitions += np.asarray(assignment) != np.asarray(previous)
                    previous = assignment
                    if changed:
                        changed_frames += 1; affected.update(changed)
                        if first is None:
                            first = item["time_fs"] / 1000.; events_context.extend(before)
                    if first is not None and item["time_fs"] / 1000. <= first + .05 + 1e-12:
                        events_context.append(item)
                    before.append(item)
                    key = (item["vacancy_mask_hungarian"], assignment)
                    if state is None or key != state["key"]:
                        if state is not None: states.append(state)
                        state = dict(key=key, first_ps=item["time_fs"] / 1000., last_ps=item["time_fs"] / 1000., frames=0,
                                     changed_atoms=changed, margins=[], distances=[], nearest_matches=0, collision_max=0, hungarian_nearest_difference_max=0)
                    state["last_ps"] = item["time_fs"] / 1000.; state["frames"] += 1
                    state["collision_max"] = max(state["collision_max"], item["nearest_collision_count"])
                    state["hungarian_nearest_difference_max"] = max(state["hungarian_nearest_difference_max"], item["hungarian_differs_from_nearest"])
                    if changed:
                        state["margins"].extend(item["nearest_second_gap_A"][i] for i in changed)
                        state["distances"].extend(item["assignment_distances_A"][i] for i in changed)
                        state["nearest_matches"] += all(item["nearest_parent_li_sites"][i] == assignment[i] for i in changed)
                if state is not None: states.append(state)
            converted = []
            for state in states:
                mask, assignment = state.pop("key"); margins = state.pop("margins"); distances = state.pop("distances")
                state.update(vacancy_mask=mask, assignment=list(assignment), sampled_residence_span_ps=state["last_ps"] - state["first_ps"],
                             is_original_assignment=assignment == baseline, same_occupation_mask_as_original=mask == baseline_mask)
                if margins:
                    state.update(affected_nearest_margin_min_A=float(np.min(margins)), affected_nearest_margin_p10_A=float(np.percentile(margins, 10)),
                                 affected_nearest_margin_median_A=float(np.median(margins)), affected_assignment_distance_max_A=float(np.max(distances)),
                                 affected_nearest_matches_assignment_fraction=state["nearest_matches"] / state["frames"])
                converted.append(state)
            if first is not None:
                changed_any_phase = True
                for item in events_context + list(before):
                    for ordinal in sorted(affected):
                        context.append(dict(identity=identity, phase=phase, time_ps=item["time_fs"] / 1000., Li_ordinal_0based=ordinal,
                            atom_index_0based=int(li_rows[ordinal]), original_parent_index_0based=int(parent_indices[li_rows[ordinal]]),
                            original_parent_Li_site_0based=baseline[ordinal], assigned_parent_Li_site_0based=item["assigned_parent_li_sites"][ordinal],
                            nearest_parent_Li_site_0based=item["nearest_parent_li_sites"][ordinal], nearest_margin_A=item["nearest_second_gap_A"][ordinal],
                            assigned_distance_A=item["assignment_distances_A"][ordinal], nearest_collisions=item["nearest_collision_count"],
                            hungarian_nearest_difference=item["hungarian_differs_from_nearest"], vacancy_mask=item["vacancy_mask_hungarian"]))
            atom_rows = []
            for ordinal in sorted(affected):
                origin = baseline[ordinal]; end = converted[-1]["assignment"][ordinal]
                _, distance = find_mic(parent_positions[parent_li_indices[end]] - parent_positions[parent_li_indices[origin]], cell, pbc=True)
                atom_rows.append(dict(Li_ordinal_0based=ordinal, atom_index_0based=int(li_rows[ordinal]), original_parent_index_0based=int(parent_indices[li_rows[ordinal]]),
                                     original_parent_Li_site_0based=origin, final_assigned_parent_Li_site_0based=end, assignment_transitions=int(transitions[ordinal]),
                                     original_to_final_parent_site_distance_A=float(distance)))
            phases[phase] = dict(frames=n, first_atom_assignment_change_ps=first,
                first_change_time_resolution_fs=receipt["detailed_interval_fs"], changed_frames=changed_frames,
                original_vacancy_mask=baseline_mask, original_canonical_vacancy_mask=canonical(baseline_mask, permutations),
                final_vacancy_mask=converted[-1]["vacancy_mask"], final_canonical_vacancy_mask=canonical(converted[-1]["vacancy_mask"], permutations),
                final_assignment_is_original=converted[-1]["is_original_assignment"],
                affected_atoms=atom_rows, state_segments=converted,
                returned_to_original_after_first_change=any(s["is_original_assignment"] and s["first_ps"] > first for s in converted) if first is not None else False,
                original_atom_assignment_stable_through10ps=first is None or first > 10.,
                source_JSONL=str(path))
            if first is not None: print(json.dumps(dict(identity=identity,phase=phase,**phases[phase])), flush=True)
        answers.append(dict(identity=identity, phases=phases, any_assignment_change=changed_any_phase))
    common10 = all((r["phases"]["equilibration"]["first_atom_assignment_change_ps"] is None and r["phases"]["production"]["original_atom_assignment_stable_through10ps"]) for r in answers if "/repeat" in r["identity"])
    result = dict(status="COMPLETED_FRAME_EVENT_AUDIT", source_sha256=sha(__file__), sources=sources,
        primary_runs_initial_assignments_stable_through10ps=common10, runs=answers,
        interpretation="Assignments are sampled every2fs. Persistent new-site residence with unambiguous nearest/Hungarian agreement supports a hop interpretation; boundary margins are reported continuously rather than thresholded. A posthoc10ps prefix is a separate diagnostic and cannot erase full-trajectory transitions.")
    (args.output / "md_audit_events.json").write_text(json.dumps(result, indent=2) + "\n")
    if context:
        unique = {(x["identity"], x["phase"], x["time_ps"], x["Li_ordinal_0based"]): x for x in context}
        rows = list(unique.values())
        with (args.output / "md_audit_event_context.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__": main()
