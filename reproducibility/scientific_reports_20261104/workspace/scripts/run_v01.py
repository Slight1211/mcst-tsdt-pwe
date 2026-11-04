"""Run exact-polygon 3D MCST comparisons in the local COMSOL environment."""
from pathlib import Path
import argparse,csv,hashlib,json,os,platform,subprocess,time,traceback
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'05_validation/V01_20261030'
NODE=ROOT/'02_evidence/scientific_data/research/wavemotion_surface_20260920/runs/node_gradient_a75_h6_N15_20261004_v1'
BIN=Path('C:/Program Files/COMSOL/COMSOL62/Multiphysics/bin/win64')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2),encoding='utf-8');t.replace(p)
def main():
    p=argparse.ArgumentParser();p.add_argument('--shape',choices=['initial','final'],required=True);p.add_argument('--mesh',type=int,required=True);p.add_argument('--layers',type=int,default=4);p.add_argument('--count',type=int,default=24);p.add_argument('--points',choices=['GX','GXM','X'],default='GX');p.add_argument('--tag',default='');args=p.parse_args()
    tag=args.tag or f'{args.shape}_M{args.mesh}L{args.layers}_n{args.count}_{args.points}'
    folder=OUT/tag;folder.mkdir(parents=True,exist_ok=False)
    for x in ['prefs','temporary']:(folder/x).mkdir()
    template=OUT/'source/ValidateShapeMcst.java';assert template.exists()
    vertices=NODE/f'{args.shape}_vertices.csv'
    settings=dict(COMSOL_MESH_LEVEL=str(args.mesh),COMSOL_DRY_SWEEP_LAYERS=str(args.layers),COMSOL_EIGEN_COUNT=str(args.count),COMSOL_EIGEN_SHIFT_MHZ='.001',COMSOL_LATTICE_UM='75',COMSOL_CORE_RADIUS_UM='5',COMSOL_RING_RADIUS_UM='22.5',COMSOL_ELECTRODE_SPLIT_UM='10',COMSOL_THICKNESS_UM='6',COMSOL_PZT_FACE_UM='0',COMSOL_RING_E_GPA='4.35',COMSOL_MCST_LENGTH_UM='1',COMSOL_NODE_GEOMETRY='false',COMSOL_ELECTRICAL_CASE='short',COMSOL_KX='0[1/m]',COMSOL_KY='0[1/m]',COMSOL_RESULT_CSV=str(folder/'spectra.csv'),COMSOL_TIMING_CSV=str(folder/'timing.csv'),MCST_VERTICES=str(vertices),MCST_RUN_DIR=str(folder),MCST_POINTS=args.points,MCST_EXPORT_FIELDS='1')
    # COMSOL's Java sandbox forbids getenv by default. Embed this case's known
    # configuration in its own source instead of granting runtime permissions.
    source=folder/'ValidateShapeMcst.java';binary=source.with_suffix('.class')
    code=template.read_text(encoding='utf-8').replace('System.getenv(name)','CASE_SETTINGS.get(name)')
    values='\n'.join('CASE_SETTINGS.put('+json.dumps(k)+','+json.dumps(v)+');' for k,v in settings.items())
    code=code.replace('public class ValidateShapeMcst {','public class ValidateShapeMcst {\nprivate static java.util.Map<String,String> CASE_SETTINGS=new java.util.HashMap<String,String>();\nstatic {\n'+values+'\n}\n')
    source.write_text(code,encoding='utf-8')
    # Preferences apply only to this case. No network or process/runtime access.
    (folder/'prefs/comsol.prefs').write_text('security.external.enable=on\nsecurity.external.filepermission=full\nsecurity.external.netpermission=off\nsecurity.external.runtimepermission=off\nsecurity.comsol.allowexternalprocess=off\n',encoding='ascii')
    with (folder/'compile.log').open('wb') as log:
        compiled=subprocess.run([str(BIN/'comsolcompile.exe'),str(source)],cwd=folder,stdout=log,stderr=subprocess.STDOUT)
    assert compiled.returncode==0 and binary.exists(),'COMSOL Java compile failed'
    command=[str(BIN/'comsolbatch.exe'),'-np','2','-prefsdir',str(folder/'prefs'),'-tmpdir',str(folder/'temporary'),'-recoverydir',str(folder/'temporary'),'-autosave','off','-inputfile',str(binary),'-nosave','-batchlog',str(folder/'batch.log')]
    manifest=dict(status='RUNNING',pid=os.getpid(),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),shape=args.shape,mesh_level=args.mesh,layers=args.layers,eigen_count=args.count,points=args.points,rtol=1e-9,integration_order=8,threads=2,source_sha256=sha(source),binary_sha256=sha(binary),vertices_sha256=sha(vertices),command=command,settings=settings,platform=platform.platform(),new_COMSOL_runs=True,human_verified=False)
    dump(folder/'manifest.json',manifest);env=os.environ.copy();env.update(settings);clock=time.perf_counter()
    try:
      with (folder/'console.log').open('wb') as log:
        result=subprocess.run(command,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT)
      manifest.update(exit_code=result.returncode,elapsed_seconds=time.perf_counter()-clock)
      rows=list(csv.DictReader((folder/'spectra.csv').open(encoding='utf-8-sig',newline=''))) if (folder/'spectra.csv').exists() else []
      count=len({r['point_index'] for r in rows});expected={'GX':2,'GXM':3,'X':1}[args.points]
      log=(folder/'batch.log').read_text(encoding='utf-8',errors='replace') if (folder/'batch.log').exists() else ''
      fields=list(folder.glob('fields_*.csv'));models=list(folder.glob('model_*.mph'))
      success=result.returncode==0 and count==expected and len(fields)==expected and len(models)==expected and 'Exception:' not in log
      manifest.update(status='COMPLETED_RAW_EXPORT' if success else 'FAILED_OR_INCOMPLETE',exported_points=count,raw_rows=len(rows),field_exports=len(fields),saved_models=len(models),scientific_validation='PENDING_BRANCH_MESH_AND_FIELD_CHECKS')
      manifest['output_sha256']={p.name:sha(p) for p in folder.glob('*') if p.is_file() and p.name!='manifest.json'}
      dump(folder/'manifest.json',manifest);print(json.dumps({k:manifest[k] for k in ['status','exit_code','elapsed_seconds','exported_points','field_exports']},indent=2),flush=True)
      if not success:raise RuntimeError('COMSOL did not produce the complete requested raw export; inspect retained logs')
    except BaseException as error:
      manifest.update(status='FAILED_OR_INCOMPLETE',error=repr(error),traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-clock);dump(folder/'manifest.json',manifest);raise
if __name__=='__main__':main()
