"""Independent scalar prediction arithmetic and raw-vector algebraic audit."""
import json
import math
import numpy as np
from threadpoolctl import threadpool_limits
import recompute as r

def main():
 r.checks();frozen=json.loads(r.require(r.FROZEN).read_text());model=json.loads(r.COEFF.read_text())
 manifest=json.loads(r.require(r.RAW/'manifest.json').read_text());assert manifest['status']=='PASS_NUMERICAL_QUALITY'
 assert frozen.get('source_variant')==manifest.get('source_variant')==r.SOURCE_VARIANT,'Audit fresh public freeze/compute outputs; historical implementation hashes do not authenticate this port'
 for path,expected in frozen['sha256'].items(): assert r.sha(path)==expected,path
 assert manifest['frozen_sha256']==r.sha(r.FROZEN)
 K0,M0=r.matrix(6.)
 with np.load(r.BASE) as d:base=d['vectors'][:,3:5].copy()
 del K0
 rows=[];maxdiff=0.
 for index,case in enumerate(frozen['cases']):
  saved=manifest['completed'][index];path=saved['source_npz'];assert r.sha(path)==saved['source_sha256']
  K,M=r.matrix(case['h_um'])
  with np.load(path) as data:
   lam=data['eigenvalues'];V=data['vectors'];freq=data['frequency_MHz']
   assert float(data['h_um'])==case['h_um'] and int(data['N'])==15
   assert np.array_equal(data['k'],r.KPOINT) and float(data['ell_epoxy_um'])==1 and float(data['ell_steel_um'])==0
   discrepancies=[];orthogonality=[]
   for seed in range(2):
    values=data['seed_eigenvalues'][seed];vectors=data['seed_vectors'][seed]
    for j in range(8):
     u=vectors[:,j];Ku=K@u;Mu=M@u
     e=float(np.linalg.norm(Ku-values[j]*Mu)/(np.linalg.norm(Ku)+abs(values[j])*np.linalg.norm(Mu)))
     assert e<1e-7,e;discrepancies.append(e)
    o=float(np.linalg.norm(vectors.T@M@vectors-np.eye(16)));assert o<1e-7;orthogonality.append(o)
   selected=[];macs=[]
   for j in range(2):
    b=base[:,j];overlaps=[]
    for k in range(8):
     v=V[:,k];overlaps.append(float(abs(b@M0@v)**2/((b@M0@b)*(v@M0@v))))
    chosen=max(range(8),key=lambda k:overlaps[k]);assert chosen==j+3
    assert overlaps[chosen]>=.99;selected.append(chosen);macs.append(overlaps[chosen])
   realf=[math.sqrt(float(lam[j]))*r.FACTOR for j in selected]
   assert max(abs(realf[j]-float(freq[selected[j]])) for j in range(2))<1e-12
   J=200*(realf[1]-realf[0])/(realf[1]+realf[0]);J0=model['baseline_X_J45_percent'];delta=J-J0
   result=dict(fractional_change=case['fractional_change'],h_um=case['h_um'],actual_f4_MHz=realf[0],actual_f5_MHz=realf[1],actual_J_percent=J,actual_delta_J_pp=delta,min_mass_MAC=min(macs),max_two_seed_residual=max(discrepancies),max_mass_orthogonality=max(orthogonality))
   for order in ['first','second']:
    predicted=[]
    for j,c in enumerate(model['coefficients']['h']):
     x=math.log1p(case['fractional_change']);exponent=c['S']*x
     if order=='second':exponent+=c['T']*x*x/2
     predicted.append(model['baseline_f4_f5_MHz'][j]*math.exp(exponent))
    jp=200*(predicted[1]-predicted[0])/(predicted[1]+predicted[0]);error=abs(jp-J)
    result.update({order+'_predicted_delta_J_pp':jp-J0,order+'_absolute_error_pp':error,order+'_increment_relative_error_percent':100*error/abs(delta),order+'_direction_agrees':(jp-J0)*delta>0})
    maxdiff=max(maxdiff,abs(jp-case[order+'_J_percent']),abs(error-saved[order+'_absolute_error_pp']))
   assert abs(J-saved['actual_J_percent'])<1e-12;rows.append(result)
  del K,M,V
 assert maxdiff<1e-12
 useful=all(v['second_absolute_error_pp']<v['first_absolute_error_pp'] and v['second_direction_agrees'] for v in rows) and max(v['second_increment_relative_error_percent'] for v in rows)<=10
 r.csvout(r.HERE/'independent_evaluation.csv',rows)
 report=dict(status='PASS_RAW_VECTOR_AND_SCALAR_AUDIT',source_variant=r.SOURCE_VARIANT,usefulness='PASS' if useful else 'FAIL',new_evaluation_points=2,independent_arithmetic_max_difference_pp=maxdiff,all_frozen_inputs_unchanged=True,rows=rows,source_sha256=r.sha(__file__),raw_manifest_sha256=r.sha(r.RAW/'manifest.json'),scope=frozen['intent'],NOT_RUN=manifest['NOT_RUN'])
 r.dump(r.HERE/'independent_audit.json',report);print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
 with threadpool_limits(limits=1):main()
