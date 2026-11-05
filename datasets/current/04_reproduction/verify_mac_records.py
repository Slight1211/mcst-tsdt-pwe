"""Check the six saved MAC records and the arithmetic of Supplementary Table S5(a)."""
from pathlib import Path
from decimal import Decimal
import argparse,csv,hashlib,json,math


def verify(root):
    root=root.resolve(); failures=[]; records={}
    with (root/'06_provenance/MAC_RECORDS.csv').open(encoding='utf-8-sig',newline='') as f:
        manifest=list(csv.DictReader(f))
    assert len(manifest)==6
    for row in manifest:
        p=(root/row['path']).resolve(); assert p.is_relative_to(root)
        data=p.read_bytes()
        if hashlib.sha256(data).hexdigest()!=row['sha256']:
            failures.append(row['case']+' '+row['grid']+' hash mismatch')
        obj=json.loads(data); records[(row['case'],int(row['nx']))]=obj
        n=int(row['nx']); nz=5 if n==21 else 7
        if obj['grid']!=[n,n,nz] or obj['sampled_points']!=n*n*nz:
            failures.append(row['case']+' sampling grid mismatch')
    with (root/'02_current_tables/Table_S5/displayed_values.csv').open(encoding='utf-8-sig',newline='') as f:
        table=[r for r in csv.DictReader(f) if r['comparison']=='solid-solid']
    computed=[]
    for row in table:
        a=records[(row['case'],21)]['MAC']; b=records[(row['case'],31)]['MAC']
        vals=[1-b[3][3],1-b[4][4],max(abs(b[j][j]-a[j][j]) for j in [3,4])]
        for field,val in zip(['branch4','branch5','sampling_change_max'],vals):
            display=Decimal(row[field]); tolerance=.5*10.0**display.as_tuple().exponent+2e-15
            if abs(val-float(display))>tolerance:
                failures.append(row['case']+' '+field+' displayed rounding mismatch')
        computed.append(dict(case=row['case'],one_minus_MAC_branch4=vals[0],one_minus_MAC_branch5=vals[1],sampling_change_max=vals[2]))
    result=dict(status='PASS' if not failures else 'FAIL',records_checked=len(manifest),table_values_checked=len(table)*3,
                failures=failures,computed=computed,raw_displacement_field_MAC_recomputed=False,new_solver_runs=0)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if not failures else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    raise SystemExit(verify(parser.parse_args().root))
