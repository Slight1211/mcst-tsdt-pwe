"""Four serial read-only exports; all15 roots and sampled-volume MAC audited."""
import json,os,shutil,subprocess,tempfile,time
from pathlib import Path
from run_joint_baseline import require_prefs
import numpy as np
from run_joint_baseline import HERE,PROJECT,BIN,preflight,sha,utc,write_json
from run_a75_cross_mesh_x import RUN,PRIOR,EXP,DEST,PREFS,verify,selected
from report_a75_joint import rows
from mode_overlap import compare

def main():
    frozen=json.loads((EXP/'inputs.json').read_text());verify(frozen)
    done=json.loads((DEST/'completion.json').read_text());assert done==json.loads((DEST/'manifest.json').read_text())
    assert done['status']=='PASS_X_SPECTRUM_REPEAT'
    assert sha(DEST/'frequencies.csv')==done['raw_sha256'] and sha(DEST/'model_A75X.mph')==done['model_sha256']
    sources={'M2L4':(DEST/'model_A75X.mph',DEST/'frequencies.csv'),
      'M1L6':(PRIOR/'checkpoints/point_4.mph',PRIOR/'frequencies_remaining.csv')}
    modes={name:[int(r['mode_index']) for r in selected(pair[1],15)] for name,pair in sources.items()}
    preflight();out=RUN/'cross_mesh_fields_v1';out.mkdir(exist_ok=False)
    scratch=Path(tempfile.mkdtemp(prefix='mcst_a75_fields_'));assert scratch.is_dir()
    for name in ('ExportJointFields.java','ExportJointFields.class','run_a75_cross_mesh_fields.py','mode_overlap.py'):shutil.copy2(HERE/name,out/name)
    rec=dict(status='RUNNING',controller_pid=os.getpid(),started_utc=utc(),mode='READ_ONLY_NO_SOLVER',
      protocol_sha256=sha(EXP/'protocol.md'),source_hashes={p.name:sha(p) for p in out.iterdir()},selected_modes=modes,exports=[])
    write_json(out/'manifest.json',rec)
    try:
        results=[]
        for n,nz in ((21,5),(31,7)):
            files=[]
            for name,(model,rawpath) in sources.items():
                verify(frozen);machine=preflight();digest=sha(model);target=out/f'{name}_{n}x{n}x{nz}.csv'
                env={k:v for k,v in os.environ.items() if not k.startswith(('COMSOL_','MCST_FIELD_'))}
                env.update(MCST_FIELD_MODEL=str(model),MCST_FIELD_CSV=str(target),MCST_FIELD_NXY=str(n),MCST_FIELD_NZ=str(nz))
                cmd=[str(BIN),'-np','4','-prefsdir',str(require_prefs(PREFS)),'-tmpdir',str(scratch),'-recoverydir',str(scratch),'-autosave','off',
                  '-inputfile',str(out/'ExportJointFields.class'),'-outputfile',str(out/f'export_{name}_{n}.mph'),'-batchlog',str(out/f'export_{name}_{n}.log')]
                item=dict(mesh=name,grid=[n,n,nz],model=str(model),model_sha256=digest,raw_path=str(rawpath),raw_sha256=sha(rawpath),command=cmd,machine=machine,status='RUNNING',started_utc=utc())
                rec['exports'].append(item);start=time.perf_counter()
                with (out/f'console_{name}_{n}.log').open('w') as log:
                    proc=subprocess.Popen(cmd,cwd=PROJECT/'comsol',env=env,stdout=log,stderr=subprocess.STDOUT)
                    item['pid']=proc.pid;write_json(out/'manifest.json',rec);item['exit_code']=proc.wait()
                assert item['exit_code']==0 and target.is_file() and sha(model)==digest
                assert 'FIELD_EXPORT_COMPLETE' in (out/f'console_{name}_{n}.log').read_text(errors='replace')
                data=np.genfromtxt(target,delimiter=',',names=True)
                assert len(data)==n*n*nz and all(np.isfinite(data[k]).all() for k in data.dtype.names)
                meta=np.genfromtxt(str(target)+'.modes.csv',delimiter=',',names=True)
                raw=[r for r in rows(rawpath) if int(r['point_index'])==4];assert len(meta)==len(raw)==15
                for r,s in zip(raw,meta):
                    assert int(r['mode_index'])==int(s['mode'])
                    for key in ('frequency_mhz','imag_frequency_mhz','kx_pi_over_a','ky_pi_over_a'):assert np.isclose(float(r[key]),s[key],rtol=1e-12,atol=1e-14),(name,key)
                item.update(status='PASS_EXPORT_METADATA',seconds=time.perf_counter()-start,csv_sha256=sha(target),metadata_sha256=sha(Path(str(target)+'.modes.csv')))
                write_json(out/'manifest.json',rec);files.append(target)
            result=dict(grid=[n,n,nz],**compare(*files,modes['M2L4'],modes['M1L6']))
            matrix=np.array(result['MAC'])
            result['edge_checks']=[dict(branch=i+1,MAC=float(matrix[i,i]),largest_row_counterpart=int(matrix[i].argmax())+1,
              passed=bool(matrix[i,i]>.95 and matrix[i].argmax()==i and np.sum(matrix[i]==matrix[i,i])==1)) for i in (3,4)]
            results.append(result);write_json(out/f'mac_{n}.json',result)
        verify(frozen);passed=all(c['passed'] for r in results for c in r['edge_checks'])
        rec.update(status='PASS_EXPORT_AND_MAC_COMPUTATION',finished_utc=utc(),edge_identity_diagnostic='PASS' if passed else 'FAIL',
          diagonal_MAC_sampling_changes=[results[1]['MAC'][i][i]-results[0]['MAC'][i][i] for i in range(6)],scope='a75,h24 X only: M2L4 versus M1L6; not full-path tracking')
        write_json(out/'manifest.json',rec);write_json(out/'completion.json',dict(**rec,comparisons=results))
        print(json.dumps(dict(status=rec['status'],edge_identity_diagnostic=rec['edge_identity_diagnostic'],edge_checks=[r['edge_checks'] for r in results]),indent=2),flush=True)
    except Exception as exc:
        rec.update(status='FAIL_EXPORT_OR_AUDIT',reason=repr(exc),finished_utc=utc())
        write_json(out/'manifest.json',rec);write_json(out/'failure.json',rec);raise

if __name__=='__main__':main()
