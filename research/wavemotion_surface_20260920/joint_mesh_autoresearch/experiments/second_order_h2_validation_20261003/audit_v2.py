"""Portable serialization wrapper; records the hashes of this public audit."""
import numpy as np
from threadpoolctl import threadpool_limits
import audit

r=audit.r
original_dump=r.dump
original_csvout=r.csvout

def plain(value):
 if isinstance(value,np.generic):return value.item()
 if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
 if isinstance(value,list):return [plain(v) for v in value]
 return value

def repaired_dump(path,report,exclusive=True):
 cleaned=plain(report)
 cleaned.update(serialization_repair='numpy bool/scalars converted to built-in types; numerical audit unchanged',wrapper_sha256=r.sha(__file__))
 report.clear();report.update(cleaned)
 original_dump(path.with_name('independent_audit_v2.json'),report,exclusive)

def repaired_csvout(path,rows):
 original_csvout(path.with_name('independent_evaluation_v2.csv'),plain(rows))

if __name__=='__main__':
 r.dump=repaired_dump;r.csvout=repaired_csvout
 with threadpool_limits(limits=1):audit.main()
