"""Small independent budget/mean checks for posthoc LOW search curves."""
import importlib.util
from pathlib import Path


PATH=Path(__file__).resolve().parents[1]/"scripts/conditioning_20261107/summarize_editor.py"
SPEC=importlib.util.spec_from_file_location("summarize_editor",PATH)
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)


def row(sid,e,novel=True,known=True):
    return dict(structure_id=sid,canonical_mask=sid,child_energy_eV=e*156,n_target=12,
                energy_known=known,novel_both=novel,passes_original_q20=True)


def test_low_curve_counts_unique_child_queries_and_retains_known_failures():
    rows=[row("known",-7,False),row("novelA",-6),row("novelA",-6),row("failed",0,known=False),row("novelB",-5)]
    curve=M.low_curves(rows,{"known"})
    assert [r["attempt"] for r in curve]==[1,2,3,4,5]
    assert [r["unique_child_queries"] for r in curve]==[1,2,2,2,3]
    assert [r["standalone_fresh_child_queries"] for r in curve]==[0,1,1,1,2]
    assert [r["best_novel_child_energy_eV_atom"] for r in curve]==["",-6.,-6.,-6.,-6.]
    assert curve[-1]["distinct_novel_original_q20_children_seen"]==2


def test_step_mean_requires_all_replicates_and_does_not_interpolate():
    a=[dict(unique_child_queries=1,best_novel_child_energy_eV_atom=""),
       dict(unique_child_queries=2,best_novel_child_energy_eV_atom=-6.),
       dict(unique_child_queries=2,best_novel_child_energy_eV_atom=-6.),
       dict(unique_child_queries=4,best_novel_child_energy_eV_atom=-8.)]
    b=[dict(unique_child_queries=1,best_novel_child_energy_eV_atom=-5.),
       dict(unique_child_queries=3,best_novel_child_energy_eV_atom=-7.)]
    mean=M.step_means([a,b],"unique_child_queries")
    assert [r["budget"] for r in mean]==[1,2,3]
    assert [r["mean_best_novel_child_energy_eV_atom"] for r in mean]==["",-5.5,-6.5]


def test_novel_minimum_is_not_confused_with_beating_the_best_known_seed():
    data=[]
    for method,energies in [("learned_edit",[-6.5,-8.]),("mutation",[-6.75,-9.])]:
        for i,e in enumerate(energies):
            data.append(dict(method=method,n_target=12,fit_seed=20261106,requested_bin="low",paired_trial_id=str(i),
                seed_scoring_id=f"seed{i}",seed_energy_eV=[-6.,-7.][i]*156,structure_id=f"{method}{i}",
                child_energy_eV=e*156,energy_known=True,novel_both=i==0))
    values,means=M.terminal_minima(data)
    assert len(values)==1 and values[0]["best_common_current_seed_energy_eV_atom"]==-7.
    assert values[0]["learned_edit_best_novel_child_energy_eV_atom"]==-6.5
    assert values[0]["learned_edit_best_child_minus_best_common_seed_meV_atom"]==500.
    assert values[0]["mutation_best_child_minus_best_common_seed_meV_atom"]==250.
    assert means[0]["fits_with_learned_novel_child_below_best_known_seed"]==0
