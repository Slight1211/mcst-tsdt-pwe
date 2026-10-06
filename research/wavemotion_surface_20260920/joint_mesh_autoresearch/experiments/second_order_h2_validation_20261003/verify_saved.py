"""Repeatable read-only independent audit; neither reruns eigensolves nor writes."""
import csv
import json
from threadpoolctl import threadpool_limits
import audit
from audit_v2 import plain

r=audit.r
saved=json.loads(r.require(r.HERE/'independent_audit_v2.json').read_text())
assert saved.get('source_variant')==r.SOURCE_VARIANT,'Re-audit public outputs; historical audit-source hashes no longer apply'
assert r.sha(r.HERE/'audit_v2.py')==saved['wrapper_sha256']

def verify_csv(path,rows):
 with (r.HERE/'independent_evaluation_v2.csv').open(newline='') as f:actual=list(csv.DictReader(f))
 assert len(rows)==len(actual)==2
 for row,record in zip(plain(rows),actual):
  for key,value in row.items():
   if isinstance(value,bool):assert record[key]==str(value)
   else:assert abs(float(record[key])-value)<1e-12,(key,value)

def verify_json(path,report,exclusive=True):
 data=plain(report)
 for key,value in data.items():assert saved[key]==value,key
 report.clear();report.update(data)

if __name__=='__main__':
 r.csvout=verify_csv;r.dump=verify_json
 with threadpool_limits(limits=1):audit.main()
