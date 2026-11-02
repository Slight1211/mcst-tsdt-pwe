"""New ell2 exact-polygon 3D MCST comparisons; archived V01 files are untouched."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse,csv,hashlib,json,os,platform,shutil,subprocess,time,traceback
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'05_validation/V01_ell2_20261031_v8'
BIN=Path('C:/Program Files/COMSOL/COMSOL62/Multiphysics/bin/win64')
OLD=ROOT/'05_validation/V01_20261030'
NODE=ROOT/'02_evidence/scientific_data/research/wavemotion_surface_20260920/runs/node_gradient_a75_h6_N15_20261004_v1'
OPT=ROOT/'05_validation/V07_ell2_reoptimization_20261030'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');t.replace(p)
def initialize():
    OUT.mkdir(exist_ok=True)
    for name in ['source','inputs']:(OUT/name).mkdir(exist_ok=True)
    inputs={'initial':NODE/'initial_vertices.csv','final':OPT/'restart/final_vertices.csv'}
    for name,p in inputs.items():
        target=OUT/'inputs'/f'{name}_vertices.csv'
        if target.exists():assert sha(target)==sha(p)
        else:shutil.copy2(p,target)
    template=OLD/'source/ValidateShapeMcst.java';target=OUT/'source/ValidateShapeMcst.java'
    if target.exists():assert sha(target)==sha(template)
    else:shutil.copy2(template,target)
    spec=OUT/'protocol.json'
    if not spec.exists():
        dump(spec,dict(status='DECLARED_BEFORE_NEW_SOLVES',date_local='2026-10-31',ell_epoxy_um=2,ell_steel_um=0,
            a_um=75,h_um=6,geometry='same saved 48 vertices; initial and newly reoptimized ell2 shape',
            baseline='M2L4, 24 eigenpairs, Gamma/X/M for both shapes',
            checks=['M3L4 at Gamma/X/M for both shapes','32 versus 24 eigenpairs at X for both shapes',
                    'M2L6 at Gamma/X for both shapes; both control locations are included'],
            target_bands=[4,5],polarization_thresholds=[.7,.8],frequency_cluster_relative_tolerance=.001,
            physical_field_grid='21x21x5',N21_PWE_comparison=True,
            PWE_ell_over_h_squared=1/9,source_template_sha256=sha(template),
            geometry_sources={name:dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for name,p in inputs.items()},
            optimizer_manifest_sha256=sha(OPT/'restart/manifest.json'),threads_per_COMSOL_process=2,
            maximum_concurrent_COMSOL_processes=2,full_BZ_gap_certified=False,human_verified=False,submission_ready=False))
def run(job):
    shape,mesh,layers,count,points=job[:5]
    overrides=job[5] if len(job)>5 else {}
    name=overrides.get('tag') or f'{shape}_M{mesh}L{layers}_n{count}_{points}';folder=OUT/name
    if (folder/'manifest.json').exists():
        m=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        assert m['status']=='COMPLETED_RAW_EXPORT',(name,m['status'])
        return dict(case=name,status=m['status'],reused_completed_case=True)
    folder.mkdir(exist_ok=False)
    for x in ['prefs','temporary']:(folder/x).mkdir()
    scratch=Path(overrides['scratch']).resolve() if 'scratch' in overrides else folder/'temporary'
    if 'scratch' in overrides:
        assert scratch.is_absolute() and str(scratch).isascii()
        scratch.mkdir(parents=True,exist_ok=False)
        (scratch/'recovery').mkdir()
    template=OUT/'source/ValidateShapeMcst.java';vertices=OUT/'inputs'/f'{shape}_vertices.csv'
    settings=dict(COMSOL_MESH_LEVEL=str(mesh),COMSOL_DRY_SWEEP_LAYERS=str(layers),COMSOL_EIGEN_COUNT=str(count),
        COMSOL_EIGEN_SHIFT_MHZ='.001',COMSOL_LATTICE_UM='75',COMSOL_CORE_RADIUS_UM='5',
        COMSOL_RING_RADIUS_UM='22.5',COMSOL_ELECTRODE_SPLIT_UM='10',COMSOL_THICKNESS_UM='6',
        COMSOL_PZT_FACE_UM='0',COMSOL_RING_E_GPA='4.35',COMSOL_MCST_LENGTH_UM='2',
        COMSOL_NODE_GEOMETRY='false',COMSOL_ELECTRICAL_CASE='short',COMSOL_KX='0[1/m]',COMSOL_KY='0[1/m]',
        COMSOL_RESULT_CSV=str(folder/'spectra.csv'),COMSOL_TIMING_CSV=str(folder/'timing.csv'),
        MCST_VERTICES=str(vertices),MCST_RUN_DIR=str(folder),MCST_POINTS=points,MCST_EXPORT_FIELDS='1')
    source=folder/'ValidateShapeMcst.java';binary=source.with_suffix('.class')
    code=template.read_text(encoding='utf-8').replace('System.getenv(name)','CASE_SETTINGS.get(name)')
    values='\n'.join('CASE_SETTINGS.put('+json.dumps(k)+','+json.dumps(v)+');' for k,v in settings.items())
    code=code.replace('public class ValidateShapeMcst {','public class ValidateShapeMcst {\nprivate static java.util.Map<String,String> CASE_SETTINGS=new java.util.HashMap<String,String>();\nstatic {\n'+values+'\n}\n')
    source.write_text(code,encoding='utf-8')
    (folder/'prefs/comsol.prefs').write_text('security.external.enable=on\nsecurity.external.filepermission=full\nsecurity.external.netpermission=off\nsecurity.external.runtimepermission=off\nsecurity.comsol.allowexternalprocess=off\n',encoding='ascii')
    command=[str(BIN/'comsolbatch.exe'),'-np','2','-prefsdir',str(folder/'prefs'),'-tmpdir',str(scratch),
             '-recoverydir',str(scratch/'recovery' if 'scratch' in overrides else scratch),'-autosave','off',
             '-inputfile',str(binary),'-nosave','-batchlog',str(folder/'batch.log')]
    manifest=dict(status='COMPILING',pid=os.getpid(),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        shape=shape,ell_um=2,mesh_level=mesh,layers=layers,eigen_count=count,points=points,rtol=1e-9,
        integration_order=8,threads=2,source_sha256=sha(source),template_sha256=sha(template),
        vertices_sha256=sha(vertices),vertices_file=vertices.relative_to(ROOT).as_posix(),command=command,
        settings=settings,platform=platform.platform(),resource_overrides=overrides,
        launcher_sha256=sha(Path(__file__)),new_COMSOL_runs=True,human_verified=False)
    dump(folder/'manifest.json',manifest);clock=time.perf_counter()
    try:
        with (folder/'compile.log').open('wb') as log:
            compiled=subprocess.run([str(BIN/'comsolcompile.exe'),str(source)],cwd=folder,stdout=log,stderr=subprocess.STDOUT)
        assert compiled.returncode==0 and binary.exists(),'COMSOL compile failed'
        manifest.update(status='RUNNING',binary_sha256=sha(binary));dump(folder/'manifest.json',manifest)
        with (folder/'console.log').open('wb') as log:
            env=os.environ.copy()
            if 'scratch' in overrides:env.update(TEMP=str(scratch),TMP=str(scratch))
            proc=subprocess.Popen(command,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT)
            manifest['COMSOL_pid']=proc.pid;dump(folder/'manifest.json',manifest)
            result=proc.wait()
        manifest.update(exit_code=result,elapsed_seconds=time.perf_counter()-clock)
        rows=list(csv.DictReader((folder/'spectra.csv').open(encoding='utf-8-sig',newline=''))) if (folder/'spectra.csv').exists() else []
        n=len({r['point_index'] for r in rows});expected={'GX':2,'GXM':3,'X':1}[points]
        log=(folder/'batch.log').read_text(encoding='utf-8',errors='replace') if (folder/'batch.log').exists() else ''
        fields=list(folder.glob('fields_*.csv'));models=list(folder.glob('model_*.mph'))
        ok=result==0 and n==expected and len(fields)==expected and len(models)==expected and 'Exception:' not in log
        manifest.update(status='COMPLETED_RAW_EXPORT' if ok else 'FAILED_OR_INCOMPLETE',
            exported_points=n,raw_rows=len(rows),field_exports=len(fields),saved_models=len(models),
            scientific_validation='PENDING_BRANCH_MESH_AND_FIELD_CHECKS')
        manifest['output_sha256']={p.name:sha(p) for p in folder.glob('*') if p.is_file() and p.name!='manifest.json'}
        dump(folder/'manifest.json',manifest)
        assert ok,'Incomplete raw export: '+name
        print('COMPLETED',name,round(manifest['elapsed_seconds'],1),flush=True)
        return dict(case=name,status=manifest['status'],elapsed_seconds=manifest['elapsed_seconds'])
    except BaseException as error:
        manifest.update(status='FAILED_OR_INCOMPLETE',error=repr(error),traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-clock)
        dump(folder/'manifest.json',manifest);raise
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['base','checks'],required=True);args=ap.parse_args()
    initialize()
    jobs=([(s,2,4,24,'GXM') for s in ['initial','final']] if args.phase=='base' else
          [(s,3,4,24,'GXM') for s in ['initial','final']]+
          [(s,2,4,32,'X') for s in ['initial','final']]+
          [(s,2,6,24,'GX') for s in ['initial','final']])
    state=dict(status='RUNNING',phase=args.phase,jobs=[],started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    path=OUT/f'queue_{args.phase}.json';dump(path,state)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run,j) for j in jobs]
        try:
            for f in as_completed(futures):
                state['jobs'].append(f.result());dump(path,state)
        except BaseException as e:
            for f in futures:f.cancel()
            state.update(status='STOPPED_ON_FAILURE',error=repr(e));dump(path,state);raise
    state['status']='COMPLETED';dump(path,state);print(json.dumps(state,indent=2),flush=True)
if __name__=='__main__':main()
