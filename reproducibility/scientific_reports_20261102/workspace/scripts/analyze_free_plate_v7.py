"""Independent high-precision Rayleigh-Lamb A0 roots and physical field checks."""
from pathlib import Path
from decimal import Decimal,localcontext
import csv,hashlib,json,math
import numpy as np
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'05_validation/V04_Rayleigh_Lamb_20261030'
E=4.35e9;nu=4.35/(2*1.59)-1;rho=1180.;h=6e-6;a=75e-6
mu=E/(2*(1+nu));lam=E*nu/((1+nu)*(1-2*nu));ct=np.sqrt(mu/rho);cl=np.sqrt((lam+2*mu)/rho)
def residual(v,k):
    # A0 is subsonic here. Decimal arithmetic avoids the cancelling zero-frequency root.
    with localcontext() as c:
        c.prec=50
        z=Decimal(str(v));kd=Decimal(str(k*h/2));ratio=Decimal(str(ct/cl))
        aa=(1-ratio**2*z**2).sqrt();bb=(1-z**2).sqrt()
        def tanh(t):
            t=(2*t).exp();return (t-1)/(t+1)
        return float((4*aa*bb*tanh(bb*kd)-(2-z*z)**2*tanh(aa*kd))/(z*z))
def root(k):
    thin=k*h/np.sqrt(6*(1-nu))
    v=brentq(lambda x:residual(x,k),.7*thin,1.1*thin,xtol=1e-14,rtol=1e-13)
    assert abs(residual(v,k))<1e-14
    return v*k*ct
def analytic_field(coords,k,omega):
    p=np.sqrt(complex((omega/cl)**2-k*k));q=np.sqrt(complex((omega/ct)**2-k*k));d=h/2
    B=-2j*k*p*np.cos(p*d)/((q*q-k*k)*np.cos(q*d))
    z=coords[:,2];ux=1j*k*np.sin(p*z)+q*B*np.sin(q*z);uz=p*np.cos(p*z)+1j*k*B*np.cos(q*z)
    return np.column_stack((ux,np.zeros(len(z)),uz))*np.exp(1j*k*coords[:,0,None])
records=[]
for folder in sorted(p for p in OUT.glob('M*L*') if (p/'manifest.json').exists()):
    meta=json.loads((folder/'manifest.json').read_text());assert meta['status']=='COMPLETED_RAW_EXPORT',folder
    raw=list(csv.DictReader((folder/'spectra.csv').open(encoding='utf-8-sig',newline='')))
    for i,qa in enumerate(meta['points']):
        k=qa*np.pi/a;omega=root(k);f=omega/(2*np.pi*1e6)
        candidates=[r for r in raw if int(r['point_index'])==i and float(r['vertical_fraction'])>.8 and float(r['frequency_mhz'])>0]
        selected=min(candidates,key=lambda r:abs(float(r['frequency_mhz'])/f-1))
        # The A0 prediction must also be the lowest positive flexural solution.
        assert selected==min(candidates,key=lambda r:float(r['frequency_mhz']))
        mode=int(selected['mode_index']);data=np.genfromtxt(folder/f'fields_q{i+1}.csv',delimiter=',',names=True)
        coords=np.column_stack([data['x_m'],data['y_m'],data['z_m']]);weights=np.repeat(data['rho_kg_m3']*data['volume_weight_m3'],3)
        # COMSOL's prescribed Floquet k uses exp(-i*k*x) for exported fields.
        # Verify that sign directly below; do not fit a spatial phase or amplitude.
        predicted=analytic_field(coords,-k,omega).reshape(-1)
        actual=np.column_stack([data[f'{d}{mode}_real']+1j*data[f'{d}{mode}_imag'] for d in ['u','v','w']]).reshape(-1)
        measured_ratio=None
        if qa<1:
            ids=np.where((abs(coords[:,1])<1e-12)&(abs(coords[:,2])<1e-12))[0]
            ids=ids[np.argsort(coords[ids,0])]
            w=data[f'w{mode}_real']+1j*data[f'w{mode}_imag']
            measured_ratio=float(np.angle(w[ids[-1]]/w[ids[0]])/(coords[ids[-1],0]-coords[ids[0],0])/k)
            assert abs(measured_ratio+1)<1e-3,measured_ratio
        overlap=abs(np.vdot(predicted,weights*actual))**2/(np.vdot(predicted,weights*predicted).real*np.vdot(actual,weights*actual).real)
        # The zone-edge folded pair has no unique eigenvector: compare its full subspace.
        cluster=[int(r['mode_index']) for r in candidates if abs(float(r['frequency_mhz'])/float(selected['frequency_mhz'])-1)<1e-3]
        B=np.column_stack([np.column_stack([data[f'{d}{b}_real']+1j*data[f'{d}{b}_imag'] for d in ['u','v','w']]).reshape(-1) for b in cluster])
        rhs=B.conj().T@(weights*predicted);gram=B.conj().T@(weights[:,None]*B)
        projection=float((rhs.conj()@np.linalg.pinv(gram,rcond=1e-12)@rhs).real/np.vdot(predicted,weights*predicted).real)
        err=100*(float(selected['frequency_mhz'])/f-1)
        assert abs(err)<1 and projection>.99,(folder,qa,err,projection)
        records.append(dict(mesh=folder.name,kx_pi_over_a=qa,k_rad_per_m=k,kh=k*h,analytic_A0_MHz=f,COMSOL_A0_MHz=float(selected['frequency_mhz']),
            signed_relative_error_percent=err,mode=mode,vertical_fraction=float(selected['vertical_fraction']),individual_MAC=float(overlap),
            cluster_modes=';'.join(map(str,cluster)),subspace_projection=projection,measured_spatial_wavevector_over_prescribed_k=measured_ratio,analytic_dispersion_residual=residual(omega/(k*ct),k),
            raw_sha256=hashlib.sha256((folder/'spectra.csv').read_bytes()).hexdigest()))
with (OUT/'Rayleigh_Lamb_A0_comparison.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
report=dict(status='PASS_CLASSICAL_FREE_FACE_A0',rows=records,c_L=cl,c_T=ct,thickness_m=h,
    equation='4*alpha*beta*tanh(beta*k*h/2)-(2-v^2)^2*tanh(alpha*k*h/2)=0; alpha=sqrt(1-(cT/cL)^2 v^2), beta=sqrt(1-v^2), v=omega/(k*cT)',
    maximum_absolute_frequency_error_percent=max(abs(r['signed_relative_error_percent']) for r in records),
    minimum_analytic_field_subspace_projection=min(r['subspace_projection'] for r in records),
    validates='homogeneous, classical ell=0 A0 free-surface branch; excludes nonzero-length MCST natural boundary conditions and two-phase interface conditions',
    human_verified=False)
(OUT/'analysis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
