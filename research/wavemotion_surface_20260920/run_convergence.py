"""Small fixed-angle convergence pilot; exact circular geometry coefficients."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,json,datetime,hashlib,time,csv,platform
import numpy as np
import scipy
from scipy.linalg import lu_factor,lu_solve,get_lapack_funcs
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'msse_20260919/python'
sys.path[:0]=[str(ROOT/'src'),str(OLD)]
from two_phase import assemble
from surface import surface_blocks,normal_selector
from acoustics import orders,fold_incidence,check_open_channels,scatter,reject_unresolved_cutoff

def fast_scatter(D,S,omega,Q,fm,fp,a):
    """Full coupled system, LU + LAPACK reciprocal 1-norm condition estimate."""
    qm,Ym,_=orders(omega,Q,**fm);qp,Yp,_=orders(omega,Q,**fp)
    reject_unresolved_cutoff(omega,Q,[fm,fp])
    if np.any(qm==0) or np.any(qp==0):raise ValueError('Exact cutoff')
    Zm,Zp=1/Ym,1/Yp
    L=D-1j*omega*S.conj().T@((Zm+Zp)[:,None]*S);F=2*S.conj().T@a
    lu,piv=lu_factor(L);u=lu_solve((lu,piv),F)
    rc,info=get_lapack_funcs('gecon',(lu,))(lu,np.linalg.norm(L,1))
    if info:raise RuntimeError('Condition estimation failed')
    r=a+1j*omega*Zm*(S@u);t=-1j*omega*Zp*(S@u)
    pi=.5*Ym.real*abs(a)**2;pr=.5*Ym.real*abs(r)**2;pt=.5*Yp.real*abs(t)**2
    if pi.sum()<=0:raise ValueError('No incident power')
    return dict(u=u,r=r,t=t,pi=pi,pr=pr,pt=pt,R=pr.sum()/pi.sum(),T=pt.sum()/pi.sum(),
        residual=np.linalg.norm(L@u-F)/(np.linalg.norm(L)*np.linalg.norm(u)+np.linalg.norm(F)),
        rcond=rc,qm=qm,qp=qp)

def main():
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=ROOT/'runs'/('convergence_'+stamp);out.mkdir()
    cfg=json.loads((ROOT/'spec_v2/Project_A_SurfaceAcoustics/configs/pilot_dimensionless.json').read_text())
    cfg['executed_pilot']={'Omega':[.2,.4,.6],'theta_deg':30,'structural_N':[3,5,7,9],
        'acoustic_at_fixed_structural_N':{'Ns':5,'Na':[0,1,2,3,5]},
        'geometry':'exact circular Bessel Fourier coefficients; no geometry quadrature',
        'runtime':'Python','purpose':'fixed-point convergence, not full frequency spectrum'}
    configtext=json.dumps(cfg,indent=2);(out/'config.json').write_text(configtext)
    meta=dict(run_id=out.name,case_id='two_phase_pilot',stage='T05_fixed_points',timestamp=stamp,
        code_commit=None,config_sha256=hashlib.sha256(configtext.encode()).hexdigest(),
        environment={'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform()},
        actually_executed=True,status='NOT_RUN',SI_mapping=None,time_convention='exp(-i*omega*t)',
        basis_normalization='cell_area_average',scaled_dof_order=['u0/a','v0/a','w/a','theta_x','theta_y'],
        surface_reference_planes=[-.05,.05],source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__),ROOT/'src/surface.py',ROOT/'src/acoustics.py',OLD/'two_phase.py',OLD/'msse_core.py']})
    rows=[];orderrows=[];checks=[];cache={};started=time.perf_counter()
    def check(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,status='PASS' if error<=tol else 'FAIL'))
    def compute(w,N,Na):
        key=(w,N,Na)
        if key in cache:return cache[key]
        start=time.perf_counter();fm=cfg['acoustic']['minus'];fp=cfg['acoustic']['plus'];h=cfg['geometry']['h_over_a']
        kp,k,inc=fold_incidence(w,fm['c'],np.pi/6);check_open_channels(w,k,Na,[fm,fp])
        s=assemble(N,k,h=h);cs=cfg['surface'];const={(side,phase):(cs['lambda_s_over_EBh']*h,cs['mu_s_over_EBh']*h) for side in [-1,1] for phase in ['A','B']}
        parts=surface_blocks(s['G']+k,h,s['F'],const)
        KA=cfg['bulk']['ell_A_over_h']**2*s['KA'];KB=cfg['bulk']['ell_B_over_h']**2*s['KB']
        D=s['Kcl']+KA+KB+sum(parts.values())-w*w*s['M']
        sel=np.max(abs(s['ij']),axis=1)<=Na;ij=s['ij'][sel];Q=(s['G']+k)[sel]
        S=normal_selector(len(s['G']))[sel];a=np.zeros(len(Q),complex);ii=np.where(np.all(ij==inc,axis=1))[0][0];a[ii]=1
        z=fast_scatter(D,S,w,Q,fm,fp,a)
        # Independent existing formulation regression at the lowest structural order.
        if N==3:
            ref=scatter(D,S,w,Q,fm,fp,a,'admittance')
            check(f'forms_{key}',max(np.linalg.norm(z[x]-ref[x])/max(np.linalg.norm(ref[x]),1e-30) for x in ['u','r','t']),1e-6)
        check(f'power_{key}',abs(z['R']+z['T']-1),1e-5);check(f'residual_{key}',z['residual'],1e-9)
        row=dict(run_id=out.name,case_id=f'w{w}_Ns{N}_Na{Na}',omega=w,Omega=w,theta_deg=30,phi_deg=0,
            kinc_x_a=kp[0],kinc_y_a=kp[1],kBZ_x_a=k[0],kBZ_y_a=k[1],incident_m=int(inc[0]),incident_n=int(inc[1]),
            medium_minus='THEORETICAL_WEAK_LOADING',medium_plus='THEORETICAL_WEAK_LOADING',formulation='impedance_LU',
            N=N,Na=Na,Pin=z['pi'].sum(),PR=z['pr'].sum(),PT=z['pt'].sum(),R_total=z['R'],T_total=z['T'],
            R_zero=z['pr'][ii]/z['pi'].sum(),T_zero=z['pt'][ii]/z['pi'].sum(),STL_total_db=-10*np.log10(z['T']),
            absorption_residual=1-z['R']-z['T'],linear_residual=z['residual'],reciprocal_condition_estimate=z['rcond'],
            min_cutoff_distance=min(np.min(abs(z['qm'])),np.min(abs(z['qp']))),
            open_orders_minus=int(np.sum(z['qm'].real>0)),open_orders_plus=int(np.sum(z['qp'].real>0)),
            diagnostic_status='PASS' if abs(z['R']+z['T']-1)<1e-5 and z['residual']<1e-9 else 'FAIL',seconds=time.perf_counter()-start)
        rows.append(row);cache[key]=row
        for j,(m,n) in enumerate(ij):
            orderrows.append(dict(case_id=row['case_id'],m=int(m),n=int(n),Kx=Q[j,0],Ky=Q[j,1],
                qminus_real=z['qm'][j].real,qminus_imag=z['qm'][j].imag,qplus_real=z['qp'][j].real,qplus_imag=z['qp'][j].imag,
                is_propagating_minus=bool(z['qm'][j].real>0),is_propagating_plus=bool(z['qp'][j].real>0),
                Pinc_order=z['pi'][j],PR_order=z['pr'][j],PT_order=z['pt'][j]))
        np.savez_compressed(out/(row['case_id']+'.npz'),Kcl=s['Kcl'],Kmc_A=KA,Kmc_B=KB,**parts,M=s['M'],S=S,
            G_indices=s['ij'],acoustic_indices=ij,k_BZ=k,N=N,Na=Na,geometry_nodes=np.empty((0,2)),
            geometry_circle_radius=np.sqrt(.3/np.pi),unit_system='dimensionless',mcst_parameters_included=True,surface_parameters_included=True,
            a=a,u=z['u'],r=z['r'],t=z['t'],qminus=z['qm'],qplus=z['qp'])
        print(row['case_id'],'T',row['T_total'],'STL',row['STL_total_db'],'seconds',row['seconds'],flush=True)
        return row
    def savecsv(path,data):
        if data:
            with path.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    try:
        conv=[]
        for w in [.2,.4,.6]:
            prev=None
            for N in [3,5,7,9]:
                cur=compute(w,N,N)
                if prev:
                    dT=abs(cur['T_total']-prev['T_total']);dB=abs(cur['STL_total_db']-prev['STL_total_db'])
                    conv.append(dict(kind='structural_and_matched_acoustic',Omega=w,previous=prev['N'],current=N,delta_T=dT,delta_STL=dB,status='PASS' if dT<=.005 and dB<=.2 else 'FAIL'))
                prev=cur
            prev=None
            for Na in [0,1,2,3,5]:
                cur=compute(w,5,Na)
                if prev:
                    dT=abs(cur['T_total']-prev['T_total']);dB=abs(cur['STL_total_db']-prev['STL_total_db'])
                    conv.append(dict(kind='acoustic_at_fixed_Ns5',Omega=w,previous=prev['Na'],current=Na,delta_T=dT,delta_STL=dB,status='PASS' if dT<=.005 and dB<=.2 else 'FAIL'))
                prev=cur
        savecsv(out/'convergence.csv',conv)
        final=[r for r in conv if (r['kind'].startswith('structural') and r['current']==9) or (r['kind'].startswith('acoustic') and r['current']==5)]
        meta['status']='PASS' if all(r['status']=='PASS' for r in final+checks) else 'FAIL'
        meta['scope']='Three fixed frequencies only; no full-spectrum convergence claimed'
    except Exception as exc:
        meta['status']='FAIL';meta['exception']=repr(exc);raise
    finally:
        meta['seconds']=time.perf_counter()-started
        savecsv(out/'scattering.csv',rows);savecsv(out/'orders.csv',orderrows)
        (out/'checks.json').write_text(json.dumps(checks,indent=2));(out/'metadata.json').write_text(json.dumps(meta,indent=2))
        print('OUTPUT',out,'STATUS',meta['status'],flush=True)
if __name__=='__main__':main()
