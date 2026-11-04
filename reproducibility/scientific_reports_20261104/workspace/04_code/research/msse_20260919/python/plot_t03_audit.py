from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'runs/t03_N13_20260919T195234489663Z'
r=json.loads((p/'refinement.json').read_text());g=json.loads((p/'geometry.json').read_text())
fig,ax=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
for row in r:ax[0].plot(range(1,13),[100*x for x in row['frequency_change']],'-o',ms=3,label=row['point'])
ax[0].axhline(.5,color='k',ls='--',lw=1,label='0.5% threshold');ax[0].set(xlabel='Ordered band index',ylabel='Frequency change (%)',title='N = 11 to 13');ax[0].legend();ax[0].grid(alpha=.2)
for point in ['X','M']:
    rows=[x for x in g if x['point']==point];ax[1].loglog([x['nodes'] for x in rows],[100*x['max_frequency_error'] for x in rows],'-o',label=point)
ax[1].set(xlabel='Polygon vertices',ylabel='Maximum frequency error (%)',title='Polygon versus analytic circle (N = 5)');ax[1].legend();ax[1].grid(alpha=.2)
fig.savefig(p/'audit.png',dpi=180)
(p/'figure_manifest.json').write_text(json.dumps(dict(data_status='computed',sources=['refinement.json','geometry.json'],script=str(Path(__file__).resolve()),limitations='Selected first12 ordered modes at X/M only; not BZ bandgap or independent FEM'),indent=2))
