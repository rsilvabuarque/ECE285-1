"""Compare saved scalar estimates for two grids of the same completed MD run."""
from pathlib import Path
import csv,hashlib,json

out=Path(__file__).resolve().parent
old=out.parent/'half_dt_control_native/native_results.csv';new=out/'native_results.csv'
def read(path):
    with path.open() as f:return {r['mode']:r for r in csv.DictReader(f)}
a,b=read(old),read(new);rows=[]
units=dict(observed_temperature_K='K',input_MD_total_mean_eV='eV/cell',
           F_classical_MD_anchored_eV='eV/cell',F_quantum_MD_anchored_eV='eV/cell',
           S_cage_quantum_eV_K='eV/(cell K)',fluidicity='dimensionless closure value',
           integrated_modes='modes/cell',saved_dt_times_Omega0='dimensionless')
for mode in a:
    for field,unit in units.items():
        x=float(a[mode][field]);y=float(b[mode][field])
        rows.append(dict(mode=mode,quantity=field,units=unit,native_observation025_value=x,
                         matched_observation05_value=y,matched_minus_native=y-x))
with (out/'observation_grid_sensitivity.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
(out/'observation_grid_comparison_receipt.json').write_text(json.dumps(dict(
    status='CALCULATED_SAME_TRAJECTORY_OBSERVATION_SENSITIVITY',
    native_csv_sha256=sha(old),matched_csv_sha256=sha(new),source_sha256=sha(__file__),
    comparison='Same completed0.25fs integration trajectory observed every0.25 or0.5fs; same5ps window/1.25ps lag',
    limitations='No causal integrator convergence-order or thermostat effect established'),indent=2)+'\n')
