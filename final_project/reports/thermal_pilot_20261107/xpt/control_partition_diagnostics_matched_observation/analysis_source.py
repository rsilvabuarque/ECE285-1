"""Recover assigned-gas mode integrals from unchanged native ATLAS spectra.

The solver fluidicity precedes the spectrum cap. The pinned atomic cage routine
instead uses f = integral(gas_DOS)/(3*N); report both, not a diffusing-atom count.
The reported gate factor is its formula evaluated on this integral, not proof
that kernel inversion/guards successfully supplied a physical cage correction.
"""
import argparse,csv,hashlib,json,os
from pathlib import Path
import numpy as np

SOURCE=Path(__file__).read_bytes()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--native-results',type=Path,nargs='+',required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    if os.environ.get('NERSC_HOST') and not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use the authorized compute allocation')
    a.output_dir.mkdir(parents=True,exist_ok=False);(a.output_dir/'analysis_source.py').write_bytes(SOURCE)
    rows=[];sources={}
    for path in a.native_results:
        sources[str(path)]=dict(native_csv_sha256=sha(path),spectra={})
        with path.open() as f:records=list(csv.DictReader(f))
        for r in records:
            spectrum=Path(r['native_prefix']).parent/'spectrum.npz'
            with np.load(spectrum,allow_pickle=False) as d:
                gas=float(np.trapezoid(d['dos_gas_per_cm_cell'],d['frequency_cm']))
                total=float(np.trapezoid(d['dos_per_cm_cell'],d['frequency_cm']))
            N=int(r.get('n_atoms') or (float(r['expected_modes'])+3)/3)
            f_gas=gas/(3*N);gate=f_gas*f_gas/(f_gas*f_gas+.01**2)
            if not np.isfinite([gas,total,f_gas,gate]).all():raise ValueError('Non-finite spectrum diagnostic')
            rows.append(dict(native_csv=str(path),basin_id=r['basin_id'],repeat_id=r['repeat_id'],mode=r['mode'],scope=r['scope'],n_atoms=N,observed_temperature_K=float(r['observed_temperature_K']),closure_fluidicity=float(r['fluidicity']),integrated_total_modes=total,integrated_assigned_gas_modes=gas,gas_fraction_of_internal_modes=gas/total,cage_f_from_gas_over_3N=f_gas,cage_gate_formula_value=gate,cage_gate_f0=.01,S_cage_quantum_eV_K=float(r['S_cage_quantum_eV_K']),minus_T_Scage_quantum_eV=-float(r['observed_temperature_K'])*float(r['S_cage_quantum_eV_K']),nonzero_reported_cage_correction=float(r['S_cage_quantum_eV_K'])!=0))
            sources[str(path)]['spectra'][str(spectrum)]=sha(spectrum)
    with (a.output_dir/'partition_diagnostics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    receipt=dict(status='CALCULATED_NATIVE_PARTITION_DIAGNOSTICS',rows=len(rows),source_sha256=hashlib.sha256(SOURCE).hexdigest(),sources=sources,units=dict(DOS_integral='modes/cell',S='eV/(cell K)',TS='eV/cell'),pinned_source='https://github.com/atlas-nano/codes/blob/197bdcf5f57c9d04b5527f9e7d6f4fdb8bf3b82d/py-xPT/src/pyxpt/thermo/utility.py#L1952',limitations=['No spectrum or thermodynamic field changed','Gas-mode fraction is an estimator assignment, not a fraction of Li atoms observed to diffuse','Gate formula value does not establish successful kernel inversion or physical cage entropy'])
    (a.output_dir/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(status=receipt['status'],rows=len(rows),output=str(a.output_dir))))

if __name__=='__main__':main()
