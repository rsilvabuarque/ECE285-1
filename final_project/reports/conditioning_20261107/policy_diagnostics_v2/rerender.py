"""Repair plot layout without changing the completed numerical tables."""
import importlib.util
import json
from pathlib import Path
import shutil


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT/"scripts/conditioning_20261107/analyze_diagnostics.py"
spec = importlib.util.spec_from_file_location("diagnostic_renderer", SOURCE)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
backup = HERE/"original_plot_layout"
backup.mkdir(exist_ok=False)
old = {}
for path in list(HERE.glob("*.png")) + list(HERE.glob("*.pdf")):
    old[path.name] = module.sha256(path)
    shutil.copyfile(path, backup/path.name)
receipt = json.loads((HERE/"receipt.json").read_text())
for name, digest in receipt["output_hashes"].items():
    assert module.sha256(HERE/name) == digest


def records(name):
    rows = module.read_csv(HERE/(name+".csv"))
    for row in rows:
        for key in ["n_Li", "optimizer_updates", "epoch", "selected_epoch", "component"]:
            if row.get(key, "") != "":row[key] = int(row[key])
        for key in ["energy_eV_atom", "mean_delta_E_eV_atom", "mean_raw_Hamming",
                    "posterior_mean_target_Hamming", "prior_mean_target_Hamming",
                    "selection_beta", "KL_nats", "posterior_mean_variance"]:
            if row.get(key, "") != "":row[key] = float(row[key])
        for key in ["energy_known", "mask_valid"]:
            if key in row:row[key] = row[key] == "True"
    return rows


module.plot_diagnostics(HERE, records("attempt_outcomes"), records("condition_summary"),
                        records("latent_summary"), records("posterior_prior_summary"))
module.plot_fit_diagnostics(HERE, records("fit_history"), records("selected_latent_diagnostics"),
                            records("latent_components"))
for name, digest in receipt["output_hashes"].items():
    assert module.sha256(HERE/name) == digest
result = dict(status="PASS_NUMERICAL_TABLES_UNCHANGED", reason="Visual review found clipped labels and an empty legend entry",
              renderer_source_sha256=module.sha256(SOURCE), rerender_source_sha256=module.sha256(__file__),
              numerical_receipt_sha256=module.sha256(HERE/"receipt.json"),
              old_plot_hashes=old, plot_hashes={p.name:module.sha256(p) for p in sorted(HERE.iterdir()) if p.suffix in {".png", ".pdf"}})
(HERE/"plot_receipt.json").write_text(json.dumps(result, indent=2)+"\n")
shutil.copyfile(SOURCE, HERE/"analyze_diagnostics.renderer.py")
print(json.dumps(result, indent=2))
