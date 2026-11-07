"""Relabel the completed NVT control using saved arrays; no numerical rerun."""
from pathlib import Path
import csv, hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

folder=Path(__file__).resolve().parent
with (folder/'native_results.csv').open() as f:
    row=next(r for r in csv.DictReader(f) if r['mode']=='3PT')
source=Path(row['native_prefix']).parent/'spectrum.npz'
with np.load(source,allow_pickle=False) as z:
    d={k:z[k] for k in z.files}
fig,axes=plt.subplots(1,2,figsize=(10,4.1))
axes[0].plot(d['frequency_THz'],d['dos_per_cm_cell']/.0299792458,label='Total DOS',linewidth=.8)
axes[0].plot(d['frequency_THz'],d['dos_gas_per_cm_cell']/.0299792458,label='Assigned gas component',linewidth=.8)
axes[0].set(xlim=(0,30),xlabel='Frequency (THz)',ylabel='Modes per THz per cell')
axes[0].legend(fontsize=8)
axes[1].plot(d['lag_ps'],d['normalized_VACF'],linewidth=.8)
axes[1].set(xlim=(0,float(row['maximum_lag_ps'])),xlabel='Lag time (ps)',ylabel='Mass-weighted VACF / VACF(0)')
for ax in axes: ax.grid(alpha=.2)
fig.suptitle(f"5 ps Bussi-NVT control; native ATLAS at {float(row['observed_temperature_K']):.3f} K\n0.5 fs saved spacing, 1.25 ps maximum lag; gas assignment does not establish diffusion",fontsize=10)
fig.tight_layout(rect=[0,0,1,.86]);fig.savefig(folder/'nvt_spectral_diagnostics.pdf');fig.savefig(folder/'nvt_spectral_diagnostics.png',dpi=180)
plt.close(fig)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
(folder/'plot_receipt.json').write_text(json.dumps(dict(source_spectrum=str(source),source_spectrum_sha256=sha(source),native_csv_sha256=sha(folder/'native_results.csv'),plot_source_sha256=sha(__file__),scope='Completed5ps NVT control only; original generic figure with incorrect NVE title preserved; numerical arrays unchanged',outputs={p.name:sha(p) for p in [folder/'nvt_spectral_diagnostics.pdf',folder/'nvt_spectral_diagnostics.png']}),indent=2)+'\n')
