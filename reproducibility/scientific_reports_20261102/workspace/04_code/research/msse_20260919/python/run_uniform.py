"""T01 primary-runtime rerun and T02 homogeneous folded-spectrum checks."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,json,datetime,hashlib,platform
import numpy as np
from numpy.polynomial import Polynomial as Poly
from numpy.polynomial.legendre import leggauss
from scipy.linalg import block_diag,eigh
from msse_core import operators,blocks
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
import check_t01_python as t01

def reference(theory,q,h,ell_h):
    # Independent polynomial displacement derivatives and full 3x3 tensors.
    # Does NOT call primary operators or thickness blocks.
    zero=Poly([0]);one=Poly([1]);z=Poly([0,1]);B=zero if theory=='FSDT' else Poly([0,0,0,-4/(3*h*h)])
    A=z+B
    P=[[one,zero,1j*q[0]*B,A,zero],[zero,one,1j*q[1]*B,zero,A],[zero,zero,one,zero,zero]]
    def deriv(v,j):return v.deriv() if j==2 else 1j*q[j]*v
    grad=[[[deriv(P[i][d],j) for d in range(5)] for j in range(3)] for i in range(3)]
    R=[[.5*(grad[2][1][d]-grad[1][2][d]) for d in range(5)],
       [.5*(grad[0][2][d]-grad[2][0][d]) for d in range(5)],
       [.5*(grad[1][0][d]-grad[0][1][d]) for d in range(5)]]
    chi=[[[.5*(deriv(R[i][d],j)+deriv(R[j][d],i)) for d in range(5)] for j in range(3)] for i in range(3)]
    E=10.;nu=.3;rho=2.;mu=E/(2*(1+nu));Q=E/(1-nu*nu);ks=5/6 if theory=='FSDT' else 1
    K=np.zeros((5,5),complex);M=K.copy()
    t,w=leggauss(8)
    for zz,ww in zip(t*h/2,w*h/2):
        pp=np.array([[f(zz) for f in row] for row in P]);g=np.array([[[f(zz) for f in row] for row in mat] for mat in grad]);c=np.array([[[f(zz) for f in row] for row in mat] for mat in chi])
        ex,ey=g[0,0],g[1,1];xy=g[0,1]+g[1,0];xz=g[0,2]+g[2,0];yz=g[1,2]+g[2,1]
        outer=lambda a,b:np.outer(a.conj(),b)
        K+=ww*(Q*(outer(ex,ex)+outer(ey,ey)+nu*(outer(ex,ey)+outer(ey,ex)))+mu*outer(xy,xy)+ks*mu*(outer(xz,xz)+outer(yz,yz))+2*mu*(ell_h*h)**2*np.einsum('ijd,ije->de',c.conj(),c))
        M+=ww*rho*pp.conj().T@pp
    return K,M

def main():
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out=ROOT/'runs'/('t02_python_'+stamp);out.mkdir(parents=True)
    tests=[];cases=[]
    def check(name,value,tol):
        tests.append(dict(name=name,value=float(value),threshold=tol,passed=bool(np.isfinite(value) and value<=tol)))
    # Rerun the full previous T01 suite against the new primary functions.
    previous_runs=set((ROOT/'runs').glob('t01_python_*'))
    t01.operators=operators;t01.blocks=blocks;t01.main()
    # The historical test harness labels its own standalone runtime as an
    # alternative; explicitly record this injected primary-source execution.
    fresh_runs=set((ROOT/'runs').glob('t01_python_*'))-previous_runs
    assert len(fresh_runs)==1
    t01_dir=fresh_runs.pop()
    provenance=json.loads((t01_dir/'metadata.json').read_text())
    provenance['runtime_name']='Python primary'
    provenance['operator_source']='python/msse_core.py (injected into historical test harness)'
    provenance['source_sha256']['python/msse_core.py']=hashlib.sha256((ROOT/'python/msse_core.py').read_bytes()).hexdigest()
    (t01_dir/'metadata.json').write_text(json.dumps(provenance,indent=2))
    G=np.array([(i,j) for i in range(-1,2) for j in range(-1,2)])*2*np.pi
    h=.1
    for theory in ['FSDT','TSDT']:
      for ell in [0.,.25]:
       for name,k in [('Gamma',[0.,0.]),('X',[np.pi,0]),('M',[np.pi,np.pi]),('interior',[.71,1.21])]:
        tag=f'{theory}_l{ell}_{name}';mat=[];mass=[];ref=[];cl=[];mc=[]
        for q in G+np.array(k):
            s=blocks(theory,q,h,'analytic');K=s[0]+ell**2*s[1];M=s[2]
            kr,mr=reference(theory,q,h,ell)
            check(tag+'_matrix_reference',np.linalg.norm(K-kr)/max(1,np.linalg.norm(K)),1e-10)
            ref.extend(eigh(kr,mr,eigvals_only=True));mat.append(K);mass.append(M);cl.append(s[0]);mc.append(s[1])
        K=block_diag(*mat);M=block_diag(*mass)
        check(tag+'_hermitian',max(np.linalg.norm(K-K.conj().T)/np.linalg.norm(K),np.linalg.norm(M-M.conj().T)/np.linalg.norm(M)),1e-10)
        np.linalg.cholesky(M)
        lam,v=eigh(K,M);reference_lam=np.sort(ref)
        check(tag+'_folded_spectrum',np.max(abs(lam-reference_lam))/max(1,np.max(abs(lam))),1e-10)
        check(tag+'_residual',np.linalg.norm(K@v-(M@v)*lam)/(np.linalg.norm(K)*np.linalg.norm(v)+np.linalg.norm(M)*np.linalg.norm(v*lam)),1e-9)
        check(tag+'_mass_normalization',np.linalg.norm(v.conj().T@M@v-np.eye(len(lam))),1e-10)
        tol=1e-10*max(1,np.max(abs(lam)));check(tag+'_negative_eigenvalue',max(0,-lam.min()-tol),0)
        nz=int(np.count_nonzero(abs(lam)<tol));check(tag+'_zero_modes',abs(nz-(3 if name=='Gamma' else 0)),0)
        ii=[5*j+d for j in range(9) for d in [0,1]];jj=[5*j+d for j in range(9) for d in [2,3,4]]
        check(tag+'_parity',max(np.linalg.norm(K[np.ix_(ii,jj)])/np.linalg.norm(K),np.linalg.norm(M[np.ix_(ii,jj)])/np.linalg.norm(M)),1e-10)
        if ell==0:check(tag+'_classical_limit',np.linalg.norm(K-block_diag(*cl)),1e-12)
        np.savez_compressed(out/(tag+'.npz'),G=G,k=k,Kcl=block_diag(*cl),Kphase=block_diag(*mc),M=M,lambda_raw=lam,modes=v,zero_flags=abs(lam)<tol)
        cases.append(dict(case=tag,ell_over_h=ell,N=1,ndof=45,zero_modes=nz,lambda_first8=lam[:8].tolist()))
    # Classical flexural long-wave check; scale variables by mass diagonal to
    # avoid losing the tiny bending eigenvalue to high rotational frequencies.
    for theory in ['FSDT','TSDT']:
        for q in [.01,.02,.04]:
            s=blocks(theory,[q,0],h,'analytic');idx=[2,3,4];K=s[0][np.ix_(idx,idx)];M=s[2][np.ix_(idx,idx)]
            D=np.diag(1/np.sqrt(np.diag(M).real));ev=eigh(D@K@D,D@M@D,eigvals_only=True)[0]
            expected=(10*h*h/(12*2*(1-.3**2)))*q**4
            check(theory+f'_low_k_{q}',abs(ev/expected-1),1e-3)
    meta=dict(project_id='MSSE-PCP',case_id='E00',run_id=out.name,task_id='T02',created_at=stamp,execution_status='completed' if all(t['passed'] for t in tests) else 'failed',validation_level='internal_consistency',model_version='full5_plane_stress_mcst_v1',runtime_name='Python primary',runtime_version=sys.version,platform=platform.platform(),code_commit=None,working_tree_dirty=None,toolboxes=[],random_seed=20260919,unit_system='a_EB_rhoB_dimensionless',time_convention='exp(-i*omega*t)',fourier_convention='exp(i*(k+G).r)',dof_order=['u0','v0','w','theta_x','theta_y'],basis_normalization='no area normalization',geometry_source='homogeneous delta Fourier coefficients',material_parameter_status='hypothetical E10 nu0.3 rho2 h0.1',config_sha256=hashlib.sha256(json.dumps(dict(E=10,nu=.3,rho=2,h=h,N=1,kpoints=['Gamma','X','M','interior']),sort_keys=True).encode()).hexdigest(),source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')})
    for name,data in [('metadata',meta),('tests',tests),('cases',cases)]: (out/(name+'.json')).write_text(json.dumps(data,indent=2))
    print('OUTPUT',out,'TESTS',len(tests),'FAILED',[t for t in tests if not t['passed']]);assert meta['execution_status']=='completed'
if __name__=='__main__':main()
