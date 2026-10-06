"""Resumable h=20 um 3D Bloch paths for section 4.1 and its mesh appendix."""
from pathlib import Path
import csv, hashlib, json, math, os, shutil, subprocess, sys, tempfile, time
from comsol_runtime import comsol_batch, preferences_file, require_file

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
MESH = ROOT / "runs/section41_mesh_20260923T125509Z"
PWE = ROOT / "runs/large_scale_3d_20260921T145045670173Z"
OUT = ROOT / "runs/section41_fullpath_M2L4_20260924"
CASES = [(t, m, l) for m, l in [(2,4),(3,4),(2,2),(2,6),(1,4)]
         for t in ("classic", "mcst")]

def validate(folder):
    rec = json.loads((folder/"manifest.json").read_text())
    assert rec.get("exit_code") == 0 and (folder/"model_Model.mph").is_file()
    rows = list(csv.DictReader((folder/"frequencies.csv").open(newline="")))
    assert sorted({int(r["point_index"]) for r in rows}) == list(range(13))
    minimum, maximum = 1.0, 0.0
    for point in range(13):
        selected = sorted((r for r in rows if int(r["point_index"]) == point
                           and float(r["frequency_mhz"]) > 1e-5
                           and float(r["vertical_fraction"]) > .8),
                          key=lambda r: float(r["frequency_mhz"]))
        assert len(selected) >= (5 if point in (0,12) else 6), (point,len(selected))
        for r in selected[:6]:
            f = float(r["frequency_mhz"])
            q = abs(float(r["imag_frequency_mhz"]))/f
            assert math.isfinite(q) and q < 1e-7
            minimum = min(minimum,float(r["vertical_fraction"]))
            maximum = max(maximum,q)
    return dict(status="PASS_EXPORT_AND_PATH_SPECTRUM",point_count=13,
                min_vertical=minimum,max_relative_imaginary=maximum)

def main():
    BIN = comsol_batch()
    profile = preferences_file(ROOT)
    for required in (PWE/"config.json", MESH/"config.json", MESH/"sources/BuildTwoPhaseValidation.class"):
        require_file(required)
    assert json.loads((PWE/"config.json").read_text())["h_um"] == 20
    assert json.loads((MESH/"config.json").read_text())["h_um"] == 20
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ("prefs","sources"): (OUT/name).mkdir(exist_ok=True)
    cfg = dict(a_um=500,h_um=20,r_over_a=.3,path_segments=4,
               cases=[f"{t}_M{m}_L{l}" for t,m,l in CASES],
               pwe_source=str(PWE),mesh_source=str(MESH),
               threads=4,time_limit_seconds=None)
    config_path = OUT/"config.json"
    if config_path.exists(): assert json.loads(config_path.read_text()) == cfg
    else: config_path.write_text(json.dumps(cfg,indent=2))
    for src in (MESH/"sources/BuildTwoPhaseValidation.java",
                MESH/"sources/BuildTwoPhaseValidation.class",Path(__file__)):
        dest = OUT/"sources"/src.name
        if dest.exists(): assert dest.read_bytes() == src.read_bytes()
        else: shutil.copy2(src,dest)
    prefs = OUT/"prefs/comsol.prefs"
    if not prefs.exists(): shutil.copy2(profile,prefs)
    scratch = Path(tempfile.gettempdir())/"mcst_section41_fullpath_scratch_20260924"
    scratch.mkdir(parents=True,exist_ok=True)
    assert scratch.is_dir()
    (OUT/"provenance.json").write_text(json.dumps(dict(
        command=sys.argv,python=sys.version,
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (OUT/"sources").iterdir()}),indent=2))
    print("OUTPUT",OUT,flush=True)
    for theory,mesh,layers in CASES:
        tag=f"{theory}_M{mesh}_L{layers}"
        folder=OUT/tag;folder.mkdir(exist_ok=True)
        try:
            print("REUSE",tag,validate(folder),flush=True)
            continue
        except (OSError,ValueError,AssertionError,KeyError): pass
        base=json.loads((MESH/tag/"manifest.json").read_text())["settings"]
        assert float(base["COMSOL_LATTICE_UM"]) == 500
        assert float(base["COMSOL_THICKNESS_UM"]) == 20
        settings=dict(base)
        settings.update(COMSOL_PATH_POINTS_PER_SEGMENT="4",
                        COMSOL_IBZ_SUBDIVISIONS="0",COMSOL_CONTROL_XM_ONLY="0",
                        COMSOL_EIGEN_COUNT="20",
                        COMSOL_RESULT_CSV=str(folder/"frequencies.csv"),
                        COMSOL_TIMING_CSV=str(folder/"timing.csv"))
        env={k:v for k,v in os.environ.items()
             if not k.startswith(("COMSOL_","VALIDATION_"))}
        env.update(settings)
        cmd=[str(BIN),"-np","4","-prefsdir",str(OUT/"prefs"),
             "-tmpdir",str(scratch),"-recoverydir",str(scratch),"-autosave","off",
             "-inputfile",str(OUT/"sources/BuildTwoPhaseValidation.class"),
             "-outputfile",str(folder/"model.mph"),"-batchlog",str(folder/"batch.log")]
        manifest=dict(status="RUNNING",settings=settings,command=cmd)
        mf=folder/"manifest.json";mf.write_text(json.dumps(manifest,indent=2))
        print("START",tag,flush=True);started=time.perf_counter()
        with (folder/"console.log").open("w") as stream:
            code=subprocess.run(cmd,cwd=PROJECT/"comsol",env=env,
                                stdout=stream,stderr=subprocess.STDOUT).returncode
        manifest.update(exit_code=code,seconds=time.perf_counter()-started)
        mf.write_text(json.dumps(manifest,indent=2))
        try: manifest.update(validate(folder))
        except (OSError,ValueError,AssertionError,KeyError) as exc:
            manifest.update(status="FAIL",reason=repr(exc))
        mf.write_text(json.dumps(manifest,indent=2))
        print("DONE",tag,manifest["status"],round(manifest["seconds"],1),flush=True)
        if manifest["status"] != "PASS_EXPORT_AND_PATH_SPECTRUM":
            raise RuntimeError(tag+": "+manifest["reason"])
    (OUT/"completion.json").write_text(json.dumps(
        dict(status="ALL_TEN_PATHS_VALIDATED"),indent=2))
    print("COMPLETE",OUT,flush=True)

if __name__ == "__main__": main()
