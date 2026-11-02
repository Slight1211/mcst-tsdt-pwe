"""Independent saved-file arithmetic and diagnostic audit of the endpoint2x2."""
from pathlib import Path
import csv,datetime,hashlib,json,math
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'05_validation/SHAPE_LENGTHSCALE_CROSS_V18_20261101'
REVIEW=ROOT/'03_review/shape_mechanism_v18'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig',newline='')))
def spectrum(p):
 r=rows(p);assert len(r)==153 and [int(v['point']) for v in r]==list(range(153))
 expected=[(i,j) for i in range(17) for j in range(i+1)]
 for v,(i,j) in zip(r,expected):
  assert abs(float(v['kx_a'])-i*math.pi/16)<1e-12 and abs(float(v['ky_a'])-j*math.pi/16)<1e-12
  f=[float(v[f'Omega{b}']) for b in range(1,7)]
  assert all(math.isfinite(x) and x>=0 for x in f) and all(x<=y+1e-9 for x,y in zip(f,f[1:]))
 L=max(float(v['Omega4']) for v in r);U=min(float(v['Omega5']) for v in r)
 lower=[int(v['point']) for v in r if abs(float(v['Omega4'])-L)<=1e-8]
 upper=[int(v['point']) for v in r if abs(float(v['Omega5'])-U)<=1e-8]
 return dict(L=L,U=U,J=200*(U-L)/(U+L),lower_tied_indices=lower,upper_tied_indices=upper),r

controller=load(OUT/'manifest.json');assert controller['status']=='PASS_FROZEN_LENGTHSCALE_2X2_N15'
bound={}
for p,h in controller['source_sha256'].items():assert sha(ROOT/p)==h;bound[p]=h
for p,h in controller['output_sha256'].items():assert sha(ROOT/p)==h;bound[p]=h
assert controller['fresh_cross_points']==306 and controller['diagonal_fresh_spotchecks']==6
cells=[];diagnostics={}
for label in ['g0_at5','g5_at0']:
 folder=OUT/label;m=load(folder/'manifest.json');assert m['status']=='PASS_FROZEN_SHAPE_LENGTHSCALE_CROSS_N15'
 assert m['N']==15 and m['ell_steel_um']==0 and not m['new_optimization'] and not m['new_COMSOL_runs']
 for mapping in ['source_sha256','input_sha256']:
  assert m[mapping]==m[mapping+'_post']
  for p,h in m[mapping].items():assert sha(ROOT/p)==h;bound[p]=h
 for name,h in m['output_sha256'].items():assert sha(folder/name)==h;bound[(folder/name).relative_to(ROOT).as_posix()]=h
 result,r=spectrum(folder/(label+'_ibz153.csv'))
 assert all(abs(result[k]-m['result'][k])<1e-11 for k in ['L','U','J'])
 assert all(result[k]==m['result'][k] for k in ['lower_tied_indices','upper_tied_indices'])
 checks=rows(folder/(label+'_checks.csv'));assert len(checks)==153
 maxres=0.;maxorth=0.
 for i,c in enumerate(checks):
  assert int(c['point'])==i
  assert c['kx_a']==r[i]['kx_a'] and c['ky_a']==r[i]['ky_a']
  residual=float(c['max_first6_nonrigid_relative_residual']);orth=float(c['mass_orthogonality_max'])
  assert residual<1e-7 and orth<1e-7;maxres=max(maxres,residual);maxorth=max(maxorth,orth)
  if i==0:assert abs(float(c['Gamma_rigid_lambda']))<1e-7
  else:assert c['Gamma_rigid_lambda']==''
 own_path=Path(m['shape_manifest']).parent/'final_ibz153.csv';own,own_r=spectrum(ROOT/own_path)
 assert sha(ROOT/own_path)==sha(OUT/(m['shape_label']+'_saved_diagonal_ibz153.csv'))
 spots=load(folder/'diagonal_spotchecks.json');assert [x['point'] for x in spots]==[0,27,136]
 for spot in spots:
  index=spot['point'];saved=[float(own_r[index][f'Omega{b}']) for b in range(1,7)]
  assert all(abs(x-y)<1e-14 for x,y in zip(saved,spot['Omega_saved']))
  start=1 if index==0 else 0
  actual=max(abs(x-y)/y for x,y in zip(spot['Omega_fresh'][start:],saved[start:]))
  assert actual<1e-7 and abs(actual-spot['max_nonrigid_relative_frequency_error'])<1e-14
  assert spot['max_first6_nonrigid_relative_residual']<1e-7 and spot['mass_orthogonality_max']<1e-7
 for ell,res,path in [(m['own_ell_um'],own,ROOT/own_path),(m['evaluation_ell_um'],result,folder/(label+'_ibz153.csv'))]:
  cells.append(dict(shape=m['shape_label'],evaluation_ell_um=ell,N=15,points=153,spectrum=path.relative_to(ROOT).as_posix(),sha256=sha(path),**res))
 diagnostics[label]=dict(max_residual=maxres,max_mass_orthogonality_error=maxorth,diagonal_spotchecks=3,spotcheck_max_relative_frequency_difference=max(x['max_nonrigid_relative_frequency_error'] for x in spots))

lookup={(c['shape'],c['evaluation_ell_um']):c['J'] for c in cells}
adapt=dict(at_ell0_g0_minus_g5_pp=lookup['g0',0.]-lookup['g5',0.],at_ell5_g5_minus_g0_pp=lookup['g5',5.]-lookup['g0',5.],at_ell0_better_shape='g0' if lookup['g0',0.]>lookup['g5',0.] else 'g5',at_ell5_better_shape='g5' if lookup['g5',5.]>lookup['g0',5.] else 'g0')
report=dict(status='PASS_INDEPENDENT_ENDPOINT_CROSS_ARITHMETIC_AND_GATES',audited_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),new_eigensolves_by_this_arithmetic_audit=0,
 actual_solver_work_separate='306 fresh off-diagonal points +6 diagonal spotchecks, all N15; no optimization or COMSOL',controller_manifest=OUT.joinpath('manifest.json').relative_to(ROOT).as_posix(),controller_sha256=sha(OUT/'manifest.json'),cells=cells,within_ell_preferences=adapt,diagnostics=diagnostics,source_and_output_hashes=bound,
 claim_limit='This tests which of the two selected saved shapes gives the larger discrete gap at each endpoint material state. It does not establish a unique global optimal shape, continuous-zone gap, a monotonic geometric trend, or a physical mechanism for circularization/notches.',human_verified=False,submission_ready=False)
REVIEW.mkdir(parents=True,exist_ok=True)
(REVIEW/'independent_endpoint_cross_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status=report['status'],cells=cells,within_ell_preferences=adapt,diagnostics=diagnostics),ensure_ascii=False,indent=2))
