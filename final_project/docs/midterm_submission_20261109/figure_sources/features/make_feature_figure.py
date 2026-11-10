"""Plot frozen featurization comparisons; no model/scorer is loaded.

Left: common first-three independent fits, macro-average over nine count/bin
cells. Right: five fit/sampling replicates, averaging three compositions per
requested bin. Error bars are sample SD, not confidence intervals. Yield is
new-to-history distinct occupation orbits in the requested class/all attempts.
"""
from pathlib import Path
import csv
import hashlib
import json
import statistics
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent

def read(name):
    with (ROOT/'inputs'/name).open(newline='') as stream:
        return list(csv.DictReader(stream))

def write(name, rows):
    with (ROOT/name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

registry = json.loads((ROOT/'source_registry.json').read_text())
for name, item in registry['inputs'].items():
    assert digest(ROOT/'inputs'/name) == item['sha256'], name

SEEDS = ['2026110801', '2026110802', '2026110803']
CONTROL = ['A_graph_cvae', 'B_spatial_ar', 'C_dense_cvae', 'D_nonphysical_ar']
LABELS = ['Graph\nCVAE', 'Spatial\nAR', 'Dense\nCVAE', 'Scrambled\nAR']
COLORS = ['#0072B2', '#D55E00', '#CC79A7', '#888888']
seed_rows = read('seed_metrics.csv')
source_means = {(r['teacher'],r['method']):r for r in read('mechanism_first3.csv')}
points, left = [], []
for method in CONTROL:
    chosen = [r for r in seed_rows if r['teacher']=='uma' and r['method']==method and r['replicate_id'] in SEEDS]
    assert sorted(r['replicate_id'] for r in chosen)==SEEDS
    values = [float(r['macro_composition_bin_yield']) for r in sorted(chosen,key=lambda r:r['replicate_id'])]
    mean, sd = statistics.mean(values), statistics.stdev(values)
    reference = source_means['uma',method]
    assert math.isclose(mean,float(reference['mean_macro_yield']),rel_tol=0,abs_tol=1e-14)
    assert math.isclose(sd,float(reference['descriptive_seed_SD']),rel_tol=0,abs_tol=1e-14)
    left.append(dict(method=method,teacher='uma',replicates=3,mean_yield_pct=100*mean,sample_SD_pp=100*sd))
    points += [dict(method=method,teacher='uma',fit_seed=seed,yield_pct=100*v) for seed,v in zip(SEEDS,values)]
write('left_common3_points.csv',points);write('left_common3_summary.csv',left)

RIGHT = ['A_radial_shells','B_radial_shells','mutation']
REQUESTS = ['low','middle','high']
all_bin = read('per_bin_summary.csv')
right = []
for method in RIGHT:
    for request in REQUESTS:
        chosen = [r for r in all_bin if r['teacher']=='uma' and r['method']==method and r['requested_bin']==request]
        assert len(chosen)==1 and int(chosen[0]['fit_or_sampler_replicates'])==5
        r=chosen[0]
        right.append(dict(method=method,teacher='uma',requested_bin=request,replicates=5,mean_yield_pct=100*float(r['mean_novel_inbin_yield']),sample_SD_pp=100*float(r['descriptive_seed_SD'])))
write('right_request5_summary.csv',right)

FIVE = ['A_graph_cvae','B_spatial_ar','A_radial_shells','B_radial_shells','mutation','uniform']
summary = {(r['teacher'],r['method']):r for r in read('summary_with_LOW_SD.csv')}
table = []
for method in FIVE:
    row=dict(method=method,replicates=5)
    for teacher in ['uma','mace']:
        r=summary[teacher,method];assert int(r['fit_or_sampler_replicates'])==5
        for source,dest in [('mean_novel_inbin_yield','balanced_mean_pct'),('descriptive_seed_SD','balanced_SD_pp'),('mean_fixed_q20_LOW_request_yield','original_q20_LOW_mean_pct'),('descriptive_seed_SD_fixed_q20_LOW_request_yield','original_q20_LOW_SD_pp')]:
            row[teacher+'_'+dest]=100*float(r[source])
    table.append(row)
write('full_fivefit_table.csv',table)

plt.rcParams.update({'font.size':7.3,'axes.titlesize':7.8,'axes.labelsize':7.5,'xtick.labelsize':7.1,'ytick.labelsize':7.0,'legend.fontsize':6.5,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'xtick.major.width':.65,'ytick.major.width':.65})
fig,axes=plt.subplots(1,2,figsize=(5.5,2.15),sharey=True)
fig.subplots_adjust(left=.085,right=.995,bottom=.22,top=.83,wspace=.20)
ax=axes[0]
for i,(row,color) in enumerate(zip(left,COLORS)):
    ax.bar(i,row['mean_yield_pct'],width=.64,color=color,alpha=.9,yerr=row['sample_SD_pp'],capsize=2,error_kw={'elinewidth':.8,'capthick':.8})
    y=[r['yield_pct'] for r in points if r['method']==row['method']]
    ax.scatter(np.array([i-.11,i,i+.11]),y,s=10,facecolors='white',edgecolors='black',linewidths=.5,zorder=3)
ax.set_xticks(range(4),LABELS);ax.set_ylabel('Novel in-bin yield (%)');ax.set_title('(a) Common three-fit controls',loc='left')
ax.set_ylim(0,80);ax.set_yticks([0,25,50,75]);ax.grid(axis='y',alpha=.14,linewidth=.5);ax.set_axisbelow(True)
ax=axes[1];colors=['#0072B2','#D55E00','#009E73'];labels=['Shell CVAE','Shell AR','Mutation'];width=.25;x=np.arange(3)
for k,(method,color,label) in enumerate(zip(RIGHT,colors,labels)):
    rows=[next(r for r in right if r['method']==method and r['requested_bin']==q) for q in REQUESTS]
    ax.bar(x+(k-1)*width,[r['mean_yield_pct'] for r in rows],width=width*.92,yerr=[r['sample_SD_pp'] for r in rows],label=label,color=color,alpha=.9,capsize=2,error_kw={'elinewidth':.75,'capthick':.75})
ax.set_xticks(x,['Low','Middle','High']);ax.set_xlabel('Requested energy class',labelpad=2);ax.set_title('(b) Five-replicate request breakdown',loc='left')
ax.grid(axis='y',alpha=.14,linewidth=.5);ax.set_axisbelow(True);ax.legend(loc='upper left',bbox_to_anchor=(.015,1.00),ncol=3,frameon=False,columnspacing=.6,handlelength=.9,handletextpad=.35,borderaxespad=0)
for suffix in ['pdf','png']:
    fig.savefig(ROOT/f'featurization_comparison.{suffix}',dpi=240)
plt.close(fig)

outputs=['left_common3_points.csv','left_common3_summary.csv','right_request5_summary.csv','full_fivefit_table.csv','featurization_comparison.pdf','featurization_comparison.png']
receipt=dict(status='PASS_SAVED_RESULT_PLOT',source_registry_sha256=digest(ROOT/'source_registry.json'),source_sha256=digest(Path(__file__)),figure_width_in=5.5,figure_height_in=2.15,outputs={name:digest(ROOT/name) for name in outputs},new_model_calls=0,new_oracle_calls=0,scope='Descriptive final performance; not a new final assessment or hypothesis selection',uncertainty='Sample SD across three common independent fits on left; five fits/sampling replicates on right, not confidence intervals')
(ROOT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
