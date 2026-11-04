"""Classical homogeneous-epoxy free-face benchmark using the same solid branch."""
from pathlib import Path
import argparse,csv,hashlib,json,os,subprocess,time,traceback
ROOT=Path(__file__).resolve().parents[1]
BIN=Path('C:/Program Files/COMSOL/COMSOL62/Multiphysics/bin/win64')
V01=ROOT/'05_validation/V01_20261030'
NODE=ROOT/'02_evidence/scientific_data/research/wavemotion_surface_20260920/runs/node_gradient_a75_h6_N15_20261004_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2),encoding='utf-8')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mesh',type=int,default=3);ap.add_argument('--layers',type=int,default=4);args=ap.parse_args()
    out=ROOT/'05_validation/V04_Rayleigh_Lamb_20261030'/f'M{args.mesh}L{args.layers}'
    out.mkdir(parents=True,exist_ok=False)
    for x in ['prefs','temporary']:(out/x).mkdir()
    source=V01/'source/ValidateShapeMcst.java'
    code=source.read_text(encoding='utf-8').replace('ValidateShapeMcst','ValidateFreeEpoxy')
    code=code.replace('"210.6[GPa]", "0.3", "7780[kg/m^3]"','"4.35[GPa]", "4.35/(2*1.59)-1", "1180[kg/m^3]"')
    assert '"210.6[GPa]"' not in code
    code=code.replace('model.title("Frozen 48-gon steel-epoxy MCST comparison");','model.title("Homogeneous epoxy classical free-plate Rayleigh-Lamb comparison");')
    start=code.index('    double[][] points = envString("MCST_POINTS"')
    end=code.index('    double solveSeconds=runValidation',start)
    code=code[:start]+'    double[][] points = new double[][]{{.2,0},{.4,0},{.6,0},{.8,0},{1.,0}};\n'+code[end:]
    start=code.index('  private static double runValidation(');end=code.index('  private static void exportFields',start)
    run=r'''
  private static double runValidation(Model m,double[][] points,String resultCsv) {
    try {
      PrintWriter writer=new PrintWriter(resultCsv);
      writer.println("point_index,kx_pi_over_a,ky_pi_over_a,mode_index,frequency_mhz,imag_frequency_mhz,vertical_fraction,partition_fraction,partition_vertical_fraction,solve_seconds");
      double total=0;
      for(int p=0;p<points.length;p++){
        m.param().set("kx",Double.toString(points[p][0])+"*pi/a");m.param().set("ky","0[1/m]");
        m.sol("sol1").clearSolutionData();long start=System.nanoTime();m.sol("sol1").runAll();
        double seconds=(System.nanoTime()-start)/1e9;total+=seconds;
        createFrequencyEvaluation(m,false);double[][] re=m.result().numerical("gevFreq").getReal(),im=m.result().numerical("gevFreq").getImag();
        for(int b=0;b<re[0].length;b++)writer.println(p+","+points[p][0]+",0,"+(b+1)+","+re[0][b]+","+im[0][b]+","+re[1][b]+","+re[2][b]+","+re[3][b]+","+seconds);
        writer.flush();String tag="q"+Integer.toString(p+1),dir=envString("MCST_RUN_DIR",".");
        m.save(dir+"/model_"+tag+".mph");exportFields(m,dir+"/fields_"+tag+".csv",13,5);
        System.out.println("FREE_PLATE_POINT_COMPLETE "+p+" seconds="+seconds);
      }
      writer.close();return total;
    }catch(Exception ex){throw new RuntimeException(ex);}
  }
'''
    code=code[:start]+run+code[end:]
    settings=dict(COMSOL_MESH_LEVEL=str(args.mesh),COMSOL_DRY_SWEEP_LAYERS=str(args.layers),COMSOL_EIGEN_COUNT='16',COMSOL_EIGEN_SHIFT_MHZ='.001',COMSOL_LATTICE_UM='75',COMSOL_CORE_RADIUS_UM='5',COMSOL_RING_RADIUS_UM='22.5',COMSOL_ELECTRODE_SPLIT_UM='10',COMSOL_THICKNESS_UM='6',COMSOL_PZT_FACE_UM='0',COMSOL_RING_E_GPA='4.35',COMSOL_MCST_LENGTH_UM='0',COMSOL_NODE_GEOMETRY='false',COMSOL_ELECTRICAL_CASE='short',COMSOL_KX='0[1/m]',COMSOL_KY='0[1/m]',COMSOL_RESULT_CSV=str(out/'spectra.csv'),COMSOL_TIMING_CSV=str(out/'timing.csv'),MCST_VERTICES=str(NODE/'initial_vertices.csv'),MCST_RUN_DIR=str(out))
    code=code.replace('System.getenv(name)','CASE_SETTINGS.get(name)')
    entries='\n'.join('CASE_SETTINGS.put('+json.dumps(k)+','+json.dumps(v)+');' for k,v in settings.items())
    code=code.replace('public class ValidateFreeEpoxy {','public class ValidateFreeEpoxy {\nprivate static java.util.Map<String,String> CASE_SETTINGS=new java.util.HashMap<String,String>();\nstatic {\n'+entries+'\n}\n')
    java=out/'ValidateFreeEpoxy.java';java.write_text(code,encoding='utf-8')
    (out/'prefs/comsol.prefs').write_text('security.external.enable=on\nsecurity.external.filepermission=full\nsecurity.external.netpermission=off\nsecurity.external.runtimepermission=off\nsecurity.comsol.allowexternalprocess=off\n',encoding='ascii')
    with (out/'compile.log').open('wb') as h:r=subprocess.run([str(BIN/'comsolcompile.exe'),str(java)],cwd=out,stdout=h,stderr=subprocess.STDOUT)
    assert r.returncode==0 and java.with_suffix('.class').exists()
    command=[str(BIN/'comsolbatch.exe'),'-np','2','-prefsdir',str(out/'prefs'),'-tmpdir',str(out/'temporary'),'-recoverydir',str(out/'temporary'),'-autosave','off','-inputfile',str(java.with_suffix('.class')),'-nosave','-batchlog',str(out/'batch.log')]
    meta=dict(status='RUNNING',pid=os.getpid(),source_sha256=sha(java),template_sha256=sha(source),command=command,mesh=args.mesh,layers=args.layers,ell_um=0,a_um=75,h_um=6,E_Pa=4.35e9,nu=4.35/(2*1.59)-1,rho=1180,points=[.2,.4,.6,.8,1.],quadratic_displacement=True,uniform_material=True,
        artificial_partition='Regular 48-gon partition retained for mesh topology; both domains have identical epoxy properties; no physical interface.',
        validates='classical free-face solid implementation only; not nonzero-length MCST surface or two-phase interface conditions',human_verified=False)
    dump(out/'manifest.json',meta);t=time.perf_counter()
    try:
        with (out/'console.log').open('wb') as h:r=subprocess.run(command,cwd=out,stdout=h,stderr=subprocess.STDOUT)
        raw=list(csv.DictReader((out/'spectra.csv').open(encoding='utf-8-sig',newline='')))
        assert r.returncode==0 and len({x['point_index'] for x in raw})==5
        fields=list(out.glob('fields_*.csv'));assert len(fields)==5
        for p in fields:
            rows=list(csv.DictReader(p.open(encoding='utf-8-sig',newline='')))
            assert max(abs(float(x['rho_kg_m3'])-1180) for x in rows)<1e-8
        meta.update(status='COMPLETED_RAW_EXPORT',exit_code=r.returncode,elapsed_seconds=time.perf_counter()-t,output_sha256={p.name:sha(p) for p in out.glob('*') if p.is_file() and p.name!='manifest.json'})
        dump(out/'manifest.json',meta);print(json.dumps({k:meta[k] for k in ['status','mesh','layers','elapsed_seconds']}),flush=True)
    except Exception as e:
        meta.update(status='FAILED',error=repr(e),traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-t);dump(out/'manifest.json',meta);raise
if __name__=='__main__':main()
