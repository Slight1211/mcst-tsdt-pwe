"""Fresh classical dry 3D COMSOL runs, matched real-material circular geometry.

Not acoustic, surface or MCST validation. Existing builder's legacy labels ignored.
"""
from pathlib import Path
import os,json,datetime,hashlib,subprocess,time,math,shutil,csv as csvlib
from comsol_runtime import comsol_bin, preferences_file, require_file
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parents[1]
BIN=Path(os.environ.get('COMSOL_BIN', '.')).expanduser().resolve()

def valid_export(path):
    if not path.exists():return False
    try:
        rows=list(csvlib.reader(line for line in path.read_text(encoding='utf-8-sig').splitlines() if line and not line.startswith('%')))
        if len(rows)!=12:return False
        for row in rows:
            values=[complex(x.replace('i','j')) for x in row]
            if not all(math.isfinite(x.real) and math.isfinite(x.imag) for x in values):return False
            if values[0].real<=0 or abs(values[0].imag)/values[0].real>1e-8:return False
            if abs(values[0]-1e6*values[1])/abs(values[0])>1e-12:return False
            if not 0<=values[2].real<=1:return False
        return True
    except (ValueError,IndexError,UnicodeError):return False

def main():
    BIN=comsol_bin()
    profile=preferences_file(ROOT)
    require_file(PROJECT/'comsol/BuildTwoPhaseValidation.class', 'compiled COMSOL builder')
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--meshes',type=int,nargs='+',default=[4,3]);args=parser.parse_args()
    out=ROOT/'runs'/('classical_comsol_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));out.mkdir()
    radius=50000*math.sqrt(.3/math.pi)
    # Batch-only profile approved by user; do not change global preferences.
    prefs=out/'prefs';prefs.mkdir()
    shutil.copy2(profile,prefs/'comsol.prefs')
    temporary=out/'temporary';temporary.mkdir()
    for mesh in args.meshes:
        name=f'classical_X_M{mesh}';csv=out/(name+'.csv')
        settings=dict(COMSOL_NODE_GEOMETRY='false',COMSOL_LATTICE_UM='50000',COMSOL_THICKNESS_UM='5000',
            COMSOL_CORE_RADIUS_UM=str(radius/2),COMSOL_RING_RADIUS_UM=str(radius),COMSOL_ELECTRODE_SPLIT_UM=str(.75*radius),
            COMSOL_RING_E_GPA='210.6',COMSOL_MCST_LENGTH_UM='0',COMSOL_PZT_FACE_UM='0',COMSOL_PZT_SWEEP_ELEMENTS='0',
            COMSOL_MCST_AUDIT_PAIR='false',COMSOL_ELECTRICAL_CASE='short',COMSOL_MESH_LEVEL=str(mesh),COMSOL_EIGEN_COUNT='12',
            COMSOL_EIGEN_SHIFT_MHZ='0.0001',COMSOL_PATH_POINTS_PER_SEGMENT='0',COMSOL_IBZ_SUBDIVISIONS='0',
            COMSOL_KX='pi/a',COMSOL_KY='0[1/m]',COMSOL_RESULT_CSV=str(csv),COMSOL_TIMING_CSV=str(out/(name+'_timing.csv')))
        env={k:v for k,v in os.environ.items() if not k.startswith(('COMSOL_','VALIDATION_'))};env.update(settings,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        command=[str(BIN/'comsolbatch.exe'),'-np','2','-prefsdir',str(out/'prefs'),'-tmpdir',str(temporary),'-recoverydir',str(temporary),'-autosave','off','-inputfile',str(PROJECT/'comsol/BuildTwoPhaseValidation.class'),
            '-outputfile',str(out/(name+'.mph')),'-batchlog',str(out/(name+'.log'))]
        meta=dict(case=name,status='NOT_RUN',actually_executed=False,command=command,settings=settings,
            materials={'matrix':{'E_Pa':4.35e9,'nu':4.35/(2*1.59)-1,'rho':1180},'inclusion':{'E_Pa':210.6e9,'nu':.3,'rho':7780}},
            source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),profile,PROJECT/'comsol/BuildTwoPhaseValidation.java',PROJECT/'comsol/BuildTwoPhaseValidation.class']},
            geometry_note='Inner/outer inclusion domains have identical steel properties; artificial partition is NOT a third phase',
            scope='Classical dry 3D solid eigenproblem at X only. No fluid, surface, MCST or piezo physics activated.')
        manifest=out/(name+'_manifest.json');manifest.write_text(json.dumps(meta,indent=2));start=time.perf_counter();print('START',name,str(out),flush=True)
        with (out/(name+'_console.log')).open('w') as stream:
            process=subprocess.run(command,cwd=PROJECT/'comsol',env=env,stdout=stream,stderr=subprocess.STDOUT)
        complete=process.returncode==0 and valid_export(csv)
        meta.update(status='PASS' if complete else 'FAIL',actually_executed=True,exit_code=process.returncode,seconds=time.perf_counter()-start)
        manifest.write_text(json.dumps(meta,indent=2));print('DONE',name,meta['status'],meta['seconds'],flush=True)
        if not complete:raise RuntimeError('COMSOL run incomplete; logs preserved')
    print('OUTPUT',out,flush=True)
if __name__=='__main__':main()
