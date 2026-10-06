"""H2: serial read-only M1L4/M1L6 X exports, audited physical-field MAC."""
import csv
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from run_joint_baseline import require_prefs
import numpy as np
from run_joint_baseline import OUT, HERE, WORK, PROJECT, BIN, preflight, sha, utc, write_json
from run_baseline_cross_mesh_x import selected
from mode_overlap import compare


def main():
    prerequisite=OUT/'cross_mesh_X_v1'
    done=json.loads((prerequisite/'completion.json').read_text())
    assert done==json.loads((prerequisite/'manifest.json').read_text())
    assert done['status']=='PASS_X_SPECTRUM_REPEAT' and done['exit_code']==0
    assert sha(prerequisite/'frequencies.csv')==done['raw_sha256']
    assert sha(Path(done['source_model']))==done['source_model_sha256']
    assert all(sha(prerequisite/name)==digest for name,digest in done['source_hashes'].items())
    protocol=WORK/'experiments/baseline_cross_mesh/protocol.md'
    assert sha(protocol)==done['protocol_sha256']
    joint=json.loads((OUT/'report_gamma_recomputed_v1/joint_audit.json').read_text())
    assert joint['status']=='PASS_DERIVED_PATH_WITH_RECOMPUTED_GAMMA'
    assert sha(OUT/'mcst_M1_L6/frequencies.csv')==joint['original_raw_sha256']
    sources={
        'M1L4': (prerequisite/'model_JointX.mph', prerequisite/'frequencies.csv'),
        'M1L6': (OUT/'mcst_M1_L6/checkpoints/point_4.mph', OUT/'mcst_M1_L6/frequencies.csv')}
    chosen={name:selected(pair[1]) for name,pair in sources.items()}
    modes={name:[int(r['mode_index']) for r in rows] for name,rows in chosen.items()}
    preflight()
    out=OUT/'cross_mesh_fields_v1'; out.mkdir(exist_ok=False)
    scratch=Path(tempfile.mkdtemp(prefix='mcst_crossmesh_export_'))
    assert scratch.is_dir()
    for name in ('ExportJointFields.java','ExportJointFields.class','run_cross_mesh_fields.py','mode_overlap.py'):
        shutil.copy2(HERE/name,out/name)
    rec=dict(status='RUNNING',controller_pid=os.getpid(),started_utc=utc(),
        mode='READ_ONLY_NO_SOLVER',protocol_sha256=sha(protocol),
        source_hashes={p.name:sha(p) for p in out.iterdir()},exports=[],selected_modes=modes)
    write_json(out/'manifest.json',rec)
    try:
        results=[]
        for n,nz in ((21,5),(31,7)):
            files=[]
            for name,(source,rawpath) in sources.items():
                machine=preflight(); digest=sha(source)
                target=out/f'{name}_{n}x{n}x{nz}.csv'
                env={k:v for k,v in os.environ.items() if not k.startswith(('COMSOL_','MCST_FIELD_'))}
                env.update(MCST_FIELD_MODEL=str(source),MCST_FIELD_CSV=str(target),
                           MCST_FIELD_NXY=str(n),MCST_FIELD_NZ=str(nz))
                cmd=[str(BIN),'-np','4','-prefsdir',str(require_prefs(OUT/'prefs')),'-tmpdir',str(scratch),
                     '-recoverydir',str(scratch),'-autosave','off','-inputfile',str(out/'ExportJointFields.class'),
                     '-outputfile',str(out/f'export_{name}_{n}.mph'),'-batchlog',str(out/f'export_{name}_{n}.log')]
                item=dict(mesh=name,grid=[n,n,nz],model=str(source),model_sha256=digest,
                    raw_sha256=sha(rawpath),command=cmd,machine=machine,status='RUNNING',started_utc=utc())
                rec['exports'].append(item); start=time.perf_counter()
                with (out/f'console_{name}_{n}.log').open('w') as log:
                    proc=subprocess.Popen(cmd,cwd=PROJECT/'comsol',env=env,stdout=log,stderr=subprocess.STDOUT)
                    item['pid']=proc.pid; write_json(out/'manifest.json',rec)
                    item['exit_code']=proc.wait()
                assert item['exit_code']==0 and target.is_file()
                assert 'FIELD_EXPORT_COMPLETE' in (out/f'console_{name}_{n}.log').read_text(errors='replace')
                assert sha(source)==digest, 'Original source model modified'
                data=np.genfromtxt(target,delimiter=',',names=True)
                assert len(data)==n*n*nz and all(np.isfinite(data[k]).all() for k in data.dtype.names)
                meta=np.genfromtxt(str(target)+'.modes.csv',delimiter=',',names=True)
                with rawpath.open(newline='') as stream:
                    raw=[r for r in csv.DictReader(stream) if int(r['point_index'])==4]
                assert len(meta)==len(raw)==20
                for r,s in zip(raw,meta):
                    assert int(r['mode_index'])==int(s['mode'])
                    for key in ('frequency_mhz','imag_frequency_mhz','kx_pi_over_a','ky_pi_over_a'):
                        assert np.isclose(float(r[key]),s[key],rtol=1e-12,atol=1e-14),(name,key)
                item.update(status='PASS_EXPORT_METADATA',finished_utc=utc(),seconds=time.perf_counter()-start,
                    csv_sha256=sha(target),metadata_sha256=sha(Path(str(target)+'.modes.csv')))
                write_json(out/'manifest.json',rec); files.append(target)
            result=dict(grid=[n,n,nz],**compare(*files,modes['M1L4'],modes['M1L6']))
            matrix=np.array(result['MAC'])
            result['edge_checks']=[dict(branch=i+1,MAC=float(matrix[i,i]),
                largest_row_counterpart=int(matrix[i].argmax())+1,
                passed=bool(matrix[i,i]>.95 and matrix[i].argmax()==i)) for i in (3,4)]
            results.append(result); write_json(out/f'mac_{n}.json',result)
        passed=all(c['passed'] for r in results for c in r['edge_checks'])
        delta=[results[1]['MAC'][i][i]-results[0]['MAC'][i][i] for i in range(6)]
        rec.update(status='PASS_EXPORT_AND_MAC_COMPUTATION',finished_utc=utc(),
            edge_identity_diagnostic='PASS' if passed else 'FAIL',diagonal_MAC_sampling_changes=delta,
            scope='Baseline X only; not all wavevectors or endpoint cases')
        write_json(out/'manifest.json',rec)
        write_json(out/'completion.json',dict(**rec,comparisons=results))
        print(json.dumps(dict(status=rec['status'],edge_identity_diagnostic=rec['edge_identity_diagnostic'],
            edge_checks=[r['edge_checks'] for r in results],diagonal_MAC_sampling_changes=delta),indent=2),flush=True)
    except Exception as exc:
        rec.update(status='FAIL_EXPORT_OR_AUDIT',reason=repr(exc),finished_utc=utc())
        write_json(out/'manifest.json',rec); write_json(out/'failure.json',rec)
        raise


if __name__=='__main__':
    main()
