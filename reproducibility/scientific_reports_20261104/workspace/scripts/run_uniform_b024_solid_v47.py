"""Run an unchanged MCST solid formulation on one exact new b=.24 boundary."""
from pathlib import Path
import argparse
import hashlib
import inspect
import json
import shutil
import numpy as np
import run_ell2_3d_v8 as archived

ROOT = Path(__file__).resolve().parents[1]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ell', type=int, choices=[1, 2], required=True)
    ap.add_argument('--kind', choices=['base', 'mesh', 'count', 'thickness'], required=True)
    args = ap.parse_args()
    summary = json.loads((ROOT / 'output/uniform_b024_v47/summary.json').read_text(encoding='utf-8'))
    case = next(c for c in summary['cases'] if c['ell_um'] == args.ell)
    assert case['bound'] == .24 and sha(ROOT / case['vertices']) == case['vertices_sha256']
    out = ROOT / f'05_validation/UNIFORM_B024_SOLID_V47_20261103/ell_{args.ell}'
    # A fixed-geometry validation is reusable only after exact coordinate-array
    # equality, not visual similarity or a frequency/gap tolerance.
    archive = ROOT / ('05_validation/V01_20261030' if args.ell == 1 else '05_validation/V01_ell2_20261031_v8')
    oldgeom = ROOT / ('02_evidence/scientific_data/research/wavemotion_surface_20260920/runs/node_gradient_a75_h6_N15_20261004_v1/final_vertices.csv' if args.ell == 1 else '05_validation/V07_ell2_reoptimization_20261030/restart/final_vertices.csv')
    identical = np.array_equal(np.loadtxt(ROOT / case['vertices'], delimiter=',', skiprows=1), np.loadtxt(oldgeom, delimiter=',', skiprows=1))
    mesh0, layers0, count0, points0 = {'base': (2, 4, 24, 'GXM'), 'mesh': (3, 4, 24, 'GXM'), 'count': (2, 4, 32, 'X'), 'thickness': (2, 6, 24, 'X' if args.ell == 1 else 'GX')}[args.kind]
    oldfolder = archive / f'final_M{mesh0}L{layers0}_n{count0}_{points0}'
    folder = out / oldfolder.name
    if (oldfolder / 'manifest.json').exists() and json.loads((oldfolder / 'manifest.json').read_text(encoding='utf-8'))['status'] != 'COMPLETED_RAW_EXPORT':
        retry = archive / (oldfolder.name + '_retry1')
        if (retry / 'manifest.json').exists():
            assert json.loads((retry / 'manifest.json').read_text(encoding='utf-8'))['status'] == 'COMPLETED_RAW_EXPORT'
            oldfolder = retry
    if identical and not folder.exists():
        oldmanifest = json.loads((oldfolder / 'manifest.json').read_text(encoding='utf-8'))
        assert oldmanifest['status'] == 'COMPLETED_RAW_EXPORT'
        assert float(oldmanifest['settings']['COMSOL_MCST_LENGTH_UM']) == args.ell
        folder.mkdir(parents=True, exist_ok=False)
        for p in oldfolder.iterdir():
            if p.is_file() and p.suffix.lower() in ['.csv', '.java', '.log', '.txt']:
                shutil.copy2(p, folder / p.name)
        binding = dict(archive_case=oldfolder.relative_to(ROOT).as_posix(),
                       archive_manifest_sha256=sha(oldfolder / 'manifest.json'),
                       archive_vertices=oldgeom.relative_to(ROOT).as_posix(), archive_vertices_sha256=sha(oldgeom),
                       selected_manifest=case['manifest'], selected_manifest_sha256=case['manifest_sha256'],
                       current_vertices_sha256=case['vertices_sha256'], exact_coordinate_array_equality=True)
        oldmanifest.update(reused_archived_export=True, new_COMSOL_runs=False, current_shape_bound=.24,
                           current_geometry_binding=binding, vertices_sha256=case['vertices_sha256'],
                           reuse_reason='Exact geometry and constitutive/mesh identity; optimization-coordinate bounds do not enter the solid model.',
                           output_sha256={p.name: sha(p) for p in folder.iterdir() if p.is_file()})
        (folder / 'manifest.json').write_text(json.dumps(oldmanifest, indent=2) + '\n', encoding='utf-8')
        print('REUSED_EXACT_GEOMETRY_SOLID_EXPORT', json.dumps(binding), flush=True)
        return
    if folder.exists():
        m = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
        if m['status'] == 'COMPLETED_RAW_EXPORT' and m.get('reused_archived_export'):
            assert m['current_geometry_binding']['current_vertices_sha256'] == case['vertices_sha256']
            print('REUSED_COMPLETED_BINDING', folder.name, flush=True)
            return
    for folder in [out, out / 'inputs', out / 'source']:
        folder.mkdir(parents=True, exist_ok=True)
    inputs = {'final': ROOT / case['vertices'],
              'initial': ROOT / '02_evidence/scientific_data/research/wavemotion_surface_20260920/runs/node_gradient_a75_h6_N15_20261004_v1/initial_vertices.csv'}
    template = ROOT / '05_validation/V01_20261030/source/ValidateShapeMcst.java'
    for src, dest in [(template, out / 'source/ValidateShapeMcst.java')] + [(p, out / 'inputs' / f'{n}_vertices.csv') for n, p in inputs.items()]:
        if dest.exists():
            assert sha(src) == sha(dest)
        else:
            shutil.copy2(src, dest)
    protocol = out / 'protocol.json'
    declaration = dict(status='DECLARED_BEFORE_SOLVES', revision='v47', ell_epoxy_um=args.ell,
                       ell_steel_um=0, shape_bound=.24, a_um=75, h_um=6,
                       selected_shape_manifest=case['manifest'], selected_shape_manifest_sha256=case['manifest_sha256'],
                       vertices_sha256=case['vertices_sha256'], source_template_sha256=sha(template),
                       equation_template='Unchanged MCST weak form and natural boundary formulation from V01.',
                       planned_cases=['M2L4 n24 Gamma/X/M', 'M3L4 n24 Gamma/X/M', 'M2L4 n32 X', 'M2L6 n24 Gamma/X'],
                       initial_controls='Existing exact regular-48-gon solid models at the same constitutive state; no optimization bound enters a fixed-geometry solid solve.',
                       human_verified=False, full_BZ_gap_certified=False)
    if protocol.exists():
        assert json.loads(protocol.read_text(encoding='utf-8')) == declaration
    else:
        protocol.write_text(json.dumps(declaration, indent=2) + '\n', encoding='utf-8')
    source = inspect.getsource(archived.run)
    assert source.count("COMSOL_MCST_LENGTH_UM='2'") == 1 and source.count('ell_um=2,') == 1
    source = source.replace("COMSOL_MCST_LENGTH_UM='2'", 'COMSOL_MCST_LENGTH_UM=str(ELL)')
    source = source.replace('ell_um=2,', 'ell_um=ELL,')
    scope = dict(archived.__dict__, OUT=out, ELL=args.ell, __file__=str(Path(__file__).resolve()))
    exec(compile(source, str(Path(__file__)) + ':archived_solid_launcher', 'exec'), scope)
    mesh, layers, count, points = {'base': (2, 4, 24, 'GXM'), 'mesh': (3, 4, 24, 'GXM'),
                                  'count': (2, 4, 32, 'X'), 'thickness': (2, 6, 24, 'GX')}[args.kind]
    scratch = f'C:/mcst_v47_scratch/ell{args.ell}_{args.kind}'
    result = scope['run'](('final', mesh, layers, count, points, {'scratch': scratch}))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
