"""Export current figures, table values and selected shapes from saved evidence.

No eigensolver, optimization or COMSOL is invoked. Tables labelled displayed
are parsed from the checked English source and retain its rounded precision.
Scientific inputs are never rewritten. The exported source map uses relative
paths and content hashes; no full manuscript or internal reviewer text is copied.
"""
from pathlib import Path
import csv, hashlib, io, json, math, re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '06_release/dataset_v76/current_payload'
REVIEW = ROOT / '03_review/dataset_release_v76'
ASSETS = '06_release/github_v20/repository/data/wave_motion_20261009_v1/assets/'
EDGE = '02_evidence/scientific_data/research/wavemotion_surface_20260920/runs/edge_focus_20260926/'
DISPLAY = []
FILES = {}
load = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
sha = lambda b: hashlib.sha256(b).hexdigest()


def source_record(display, dest, source, mode, note):
    DISPLAY.append(dict(display=display, path=dest, source_path=source,
                        source_sha256=sha((ROOT / source).read_bytes()),
                        export_mode=mode, units_and_scope=note))


def put(dest, data):
    assert not re.search(r'C:[/\\]+Users[/\\]+Sligh', data.decode('utf-8', errors='ignore'), re.I), dest
    assert dest not in FILES, dest
    FILES[dest] = data


def copy(display, dest, source, note):
    put(dest, (ROOT / source).read_bytes())
    source_record(display, dest, source, 'byte-identical copy', note)


def jwrite(display, dest, obj, source, note):
    put(dest, (json.dumps(obj, ensure_ascii=False, indent=2) + '\n').encode())
    source_record(display, dest, source, 'field selection / derived metadata; numerical values unchanged', note)


def cwrite(display, dest, rows, source, note, fields=None):
    rows = list(rows)
    assert rows, dest
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    put(dest, stream.getvalue().encode('utf-8-sig'))
    source_record(display, dest, source, 'selected/parsed rows; no numerical solve', note)


def readme(folder, text):
    put(folder + '/README.txt', (text.rstrip() + '\n').encode())


def csvrows(source):
    with (ROOT / source).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def table_rows(text):
    # The current tables contain no nested tabulars or escaped ampersands.
    body = re.search(r'\\begin\{tabularx?\}.*?\\toprule(.*?)\\end\{tabularx?\}', text, re.S).group(1)
    body = re.sub(r'\\(?:toprule|midrule|bottomrule)', '', body)
    body = re.sub(r'\\cmidrule(?:\([^)]*\))?\{[^}]*\}', '', body)
    return [[v.strip() for v in row.strip().split('&')] for row in body.split(r'\\') if row.strip()]


def number(cell):
    value = cell.strip().replace('$', '').replace(' ', '')
    value = re.sub(r'\\times10\^\{(-?\d+)\}', r'e\1', value)
    value = re.sub(r'10\^\{(-?\d+)\}', r'1e\1', value)
    assert re.fullmatch(r'[<>]?[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?', value), cell
    return value


def txt(cell):
    return cell.replace('$', '').replace(r'\Gamma', 'Gamma').replace(r'\ell', 'ell').replace(r'\mu\mathrm m', 'um').strip()


def export_tables():
    blocks = {}
    for doc, prefix in [('main', ''), ('supplementary', 'S')]:
        src = f'latex/{doc}.tex'
        for i, block in enumerate(re.findall(r'\\begin\{table\}.*?\\end\{table\}', (ROOT/src).read_text(encoding='utf-8'), re.S), 1):
            key = prefix + str(i)
            blocks[key] = (src, table_rows(block))
    assert set(blocks) == {'1','2','S1','S2','S3','S4','S5','S6','S7'}
    for key, (src, rr) in blocks.items():
        folder = f'02_current_tables/Table_{key}'
        # Preserve exact cells as an audit-friendly table-only transcription.
        cwrite('Table '+key, folder+'/display_cells.csv',
               [dict(row=i+1, **{f'cell_{j+1}':r[j] if j<len(r) else '' for j in range(max(map(len,rr)))}) for i,r in enumerate(rr)],
               src, 'Exact displayed cells, including LaTeX symbols; values are rounded as in the article.')
        data=[]
        if key == '1':
            quantities=['frequency','R_mc_equals_S_ell','R_sh','kappa_minus1','kappa_3','eta_3','S_h','dJ_dln_ell','dJ_dln_h']
            assert len(rr)==10
            for q,r in zip(quantities,rr[1:]):
                if len(r)==3:
                    row=dict(quantity=q,mode4=number(r[1]),mode5=number(r[2]),mode_pair='',unit='MHz' if q=='frequency' else 'dimensionless')
                else:
                    match=re.fullmatch(r'\\multicolumn\{2\}\{c\}\{\$(.*?)\$\}',r[1]);assert match
                    row=dict(quantity=q,mode4='',mode5='',mode_pair=number(match[1]),unit='percentage points per unit ln(parameter)')
                data.append(row)
        elif key == '2':
            for r in rr[1:]:
                data.append(dict(ell_um=number(r[0]),model=txt(r[1]),measure=txt(r[2]),initial_gap_percent=number(r[3]),optimized_gap_percent=number(r[4]),gain_pp=number(r[5]),points=number(r[6])))
        elif key == 'S1':
            for r in rr[1:]:
                data.append(dict(parameter=txt(r[0]),change_percent=number(r[1]),mode=number(r[2]),predicted_delta_ln_f=number(r[3]),solved_delta_ln_f=number(r[4]),absolute_error=number(r[5]),MAC=number(r[6])))
        elif key == 'S2':
            for r in rr[1:]:
                data.append(dict(hexahedra=number(r[0]),max_relative_frequency_error=number(r[1]),curvature_increment_relative_error=number(r[2]),max_constraint_discrepancy_percent=number(r[3])))
        elif key == 'S3':
            for r in rr[1:]:
                data.append(dict(k_a_over_pi=number(r[0]),analytic_frequency_MHz=number(r[1]),M3L4_signed_error_percent=number(r[2]),M2L6_signed_error_percent=number(r[3])))
        elif key == 'S4':
            geometry=None
            for r in rr:
                if len(r)==1:
                    match=re.search(r'\(a,h\)=\(([^,]+),([^)]*)\)',r[0]);assert match
                    geometry=match.groups()
                elif r[0].startswith('M'):
                    assert geometry
                    data.append(dict(a_um=geometry[0],h_um=geometry[1],solid_mesh=r[0],FSDT_lower_error_percent=number(r[1]),FSDT_upper_error_percent=number(r[2]),TSDT_lower_error_percent=number(r[3]),TSDT_upper_error_percent=number(r[4])))
        elif key == 'S5':
            kind=None;case=None
            for r in rr:
                if len(r)==1:
                    kind='solid-solid' if '(a)' in r[0] else 'PWE-solid'
                elif r[0]=='Case' or 'Plate model' in r:
                    continue
                else:
                    case=txt(r[0]) if r[0] else case
                    data.append(dict(comparison=kind,case=case,models=txt(r[1]),metric='1-MAC' if kind=='solid-solid' else 'MAC',branch4=number(r[2]),branch5=number(r[3]),sampling_change_max=number(r[4])))
        elif key == 'S6':
            for r in rr[1:]:
                value=number(r[3]);comparator='<' if value.startswith('<') else '='
                data.append(dict(ell_um=number(r[0]),quantity=txt(r[1]),coverage=txt(r[2]),comparator=comparator,value=value.lstrip('<'),unit='dimensionless' if 'projection' in r[1] else 'percent'))
        elif key == 'S7':
            for r in rr[1:]:
                data.append(dict(ell_um=number(r[0]),relative_gap_percent=number(r[1]),lower_Omega=number(r[2]),upper_Omega=number(r[3]),max_abs_shape_coordinate=number(r[4]),bound_active_coordinates=number(r[5])))
        expected={'1':9,'2':4,'S1':16,'S2':2,'S3':5,'S4':11,'S5':9,'S6':10,'S7':5}[key]
        assert len(data)==expected,(key,len(data))
        cwrite('Table '+key,folder+'/displayed_values.csv',data,src,'Machine-readable transcription of the displayed table. Units are stated in column names; display-rounded precision, not raw solver output.')
        note='Current table values parsed from the checked article. displayed_values.csv has normalized numeric fields; display_cells.csv retains exact table cells and symbols. These files reproduce reported precision, not unrounded solver values.'
        if key=='S5':
            note+='\nThe solid-solid rows are reported-value transcriptions. Complete unrounded source files for every solid-solid sampling comparison were not located; these transcriptions are not a new independent recalculation.'
        readme(folder,note)


def main():
    assert not OUT.exists(), 'Use a new destination or inspect previous export; never silently overwrite.'
    export_tables()
    figure_names={
        '1':'figure1_latexfont_compact20261005','2':'figure2_baseline_v43','3':'figure2_scans_v39',
        '4':'figure4_energy_compact_v40','5':'figure5_optimization_referee_v61','6':'figure6_shapes_v75',
        'S1':'figureS1_convergence_v31','S2':'figureS2_representative_v31','S3':'figureB1_latexfont20261005'}
    for key,stem in figure_names.items():
        copy('Figure '+key,f'01_current_figures/Figure_{key}/figure.pdf',f'latex/figures/{stem}.pdf','Current vector artwork; no raster resampling or figure modification.')
    readme('01_current_figures/Figure_1','Geometry and wavevector convention schematic, not measured or solved response data. The inclusion radius is r/a=0.30. The Brillouin-zone path is Gamma-X-M-Gamma. No numerical dataset is needed to interpret this schematic.')
    for dest,src in [
        ('six_model_spectra.csv','03_review/figures_v31/baseline/figure1_spectra_v31.csv'),
        ('path_band_edges.csv',EDGE+'section41_path_edges.csv'),
        ('plotted_edge_discrepancies.csv','03_review/figure2_labels_v43/edge_error_records.csv')]:
        copy('Figure 2','01_current_figures/Figure_2/'+dest,src,'Large-cell six-model spectra / matching-constitutive solid band-edge comparisons; units follow headers.')
    readme('01_current_figures/Figure_2','Large-cell comparison: a=500 um, h=20 um, r=150 um. Plate N=21; solid M2L4. Each plate-model discrepancy uses the solid model with the same constitutive law. Six model spectra are retained even though only selected curves appear in panel a.')
    for dest,src in [
        ('physical_gaps.csv','03_review/figures_v31/physical/physical_gap_records.csv'),
        ('size_scan_vs_MCST_solid.csv',EDGE+'scale_all_vs_3d_mcst_10scales.csv'),
        ('thickness_scan_vs_MCST_solid.csv',EDGE+'thickness_all_vs_3d_mcst.csv')]:
        copy('Figure 3','01_current_figures/Figure_3/'+dest,src,'N=15 scans. All plate models use the same-geometry MCST-M2L4 solid reference; units follow headers.')
    readme('01_current_figures/Figure_3','Cell-size and thickness scans. Grey/classical and blue/MCST plate models are compared with the common MCST solid model. Percent frequency errors and percentage-point gap differences are distinct quantities. Source records can contain additional unused metadata columns.')
    copy('Figure 4','01_current_figures/Figure_4/modal_fields.npz',ASSETS+'item-010.npz','Saved 512x512 periodic amplitude and normalized modal-energy fields; modes 4 and 5 at fixed X.')
    copy('Figure 4g','01_current_figures/Figure_4/first_order_checks.csv','03_review/sensitivity_v26/plot_source.csv','Sixteen fixed-X first-order frequency comparisons; natural-log changes dimensionless, baseline/direct frequencies in MHz.')
    readme('01_current_figures/Figure_4','modal_fields.npz stores w0/w1 (phase-aligned normalized periodic amplitudes), mc0/mc1 (couple-stress density divided by area-mean total potential energy), shear0/shear1 (classical shear density with the same normalization). Suffixes 0 and 1 mean modes 4 and 5. Grid x/a,y/a=arange(512)/512-0.5. Steel has radius 0.30 and zero couple-stress density; shear is not masked there. first_order_checks.csv supports panel g; the plotted residual is direct minus prediction, opposite to signed_log_error_prediction_minus_solve in the CSV.')
    result_source='output/uniform_b024_v47/current_results.json'
    results=load(ROOT/result_source)
    cases=sorted(results['cases'],key=lambda c:c['ell_um'])
    assert [c['ell_um'] for c in cases]==[0,.5,1,2,5]
    case1=next(c for c in cases if c['ell_um']==1)
    regular=next(s for s in case1['starts'] if s['seed']=='regular')
    folder=Path(regular['manifest']).parent.as_posix()
    copy('Figure 5','01_current_figures/Figure_5/regular_start_trajectory.json',folder+'/trajectory.json','Saved SLSQP callback history, regular-start ell=1 design; objective J in percent, dimensionless shape coordinates.')
    # The accepted terminal state differs from a rejected last callback; export only numerical fields needed to reproduce the plotted append rule.
    meta=load(ROOT/regular['manifest'])
    terminal=dict(design=meta['design'],final_ibz153=meta['final_ibz153'],optimizer=meta['rounds'][-1]['optimizer'])
    jwrite('Figure 5','01_current_figures/Figure_5/accepted_terminal_state.json',terminal,regular['manifest'],'Exact selected terminal numerical fields from the regular-start manifest; use together with callback trajectory.')
    initial='02_evidence/scientific_data/research/wavemotion_surface_20260920/runs/node_gradient_a75_h6_N15_20261004_v1/initial_ibz45.csv'
    final45=(Path(case1['manifest']).parent/'final_ibz45.csv').as_posix()
    for dest,src in [('initial_ibz45.csv',initial),('selected_final_ibz45.csv',final45),('regular_start_seed_vertices.csv',folder+'/seed_vertices.csv')]:
        copy('Figure 5','01_current_figures/Figure_5/'+dest,src,'Geometry x/a,y/a and dimensionless Omega spectra; see column headers.')
    readme('01_current_figures/Figure_5','Panels a-c use the regular-start trajectory at ell=1 um. Preserve each callback; append accepted_terminal_state if its design or active constraint violation differs from the final callback, at the same final iteration coordinate. Panel d uses the selected two-start optimized geometry and the boundary of the 45-point set. Normalize f_MHz=Omega*sqrt(4.35e9/1180)/(75e-6*2*pi*1e6). Figure 5 uses terminal/active-constraint labelling of the current article.')
    summary=[]
    for case in cases:
        assert case['N']==15 and case['bound']==.24 and case['points']==153
        ell=case['ell_um'];folder='03_optimized_shapes/ell_'+f'{ell:g}'.replace('.','p')
        for kind,dest in [('vertices','vertices_xa_ya.csv'),('spectrum','spectrum_ibz153.csv')]:
            assert sha((ROOT/case[kind]).read_bytes())==case[kind+'_sha256']
            copy('Figure 6; Table S7',folder+'/'+dest,case[kind],'48 vertices normalized by cell size a; 153-point spectrum uses dimensionless Omega. Files copied byte-for-byte.')
        values={k:case[k] for k in ['ell_um','bound','N','points','J153_percent','L','U','kL','kU','design','max_abs_design','bound_active_variables','radial_ratio','projected_stationarity','complementarity','max_constraint_violation','two_start_spread_pp']}
        values.update(a_um=75,h_um=6,steel_area_fraction=0.09*math.pi,selected_source_manifest=case['manifest'],selected_source_manifest_sha256=case['manifest_sha256'])
        jwrite('Figure 6; Table S7',folder+'/metadata.json',values,result_source,'Current selected own-length-scale design only; dimensionless k*a, Omega and shape coordinates; local six-variable result.')
        rr=csvrows(case['spectrum']);assert len(rr)==153
        lower=max(float(r['Omega4']) for r in rr);upper=min(float(r['Omega5']) for r in rr)
        gap=200*(upper-lower)/(upper+lower)
        assert abs(gap-case['J153_percent'])<1e-9
        assert len(csvrows(case['vertices']))==48
        summary.append({k:values[k] for k in ['ell_um','bound','N','points','J153_percent','L','U','radial_ratio','max_abs_design','bound_active_variables','projected_stationarity','complementarity','max_constraint_violation','two_start_spread_pp']})
    cwrite('Figure 6; Table S7','03_optimized_shapes/summary.csv',summary,result_source,'Five current selected N15 b=0.24 designs; gap percent and dimensionless convergence quantities.')
    copy('Figure 6','03_optimized_shapes/initial_regular_48gon.csv','05_validation/UNIFORM_B024_V47_20261103/ell_1_regular/seed_vertices.csv','Common equal-area initial regular 48-gon; x/a and y/a.')
    readme('01_current_figures/Figure_6','Current five separately optimized interfaces are in ../../03_optimized_shapes/. Use summary.csv, per-ell vertices and common initial_regular_48gon.csv. All use b=0.24, N=15, 153 points, fixed a=75 um, h=6 um and steel fraction 0.09*pi. No transferred-shape or endpoint cross-evaluation result is a current Figure 6 panel.')
    copy('Figure S1','01_current_figures/Figure_S1/Fourier_order_scan.csv',ASSETS+'item-053.csv','Circular reference cell Fourier-order convergence N=7..21; MHz, percent and percentage-point quantities stated in headers.')
    readme('01_current_figures/Figure_S1','Circular-cell Fourier-order scan; this is not optimized-shape N=21 evidence. N=21 is the finite-order comparison, not an exact analytical result.')
    copy('Figure S2','01_current_figures/Figure_S2/representative_spectra.csv','03_review/figures_v31/physical/representative_spectra_records.csv','N15 representative circular-cell spectra; dimensionless Omega.')
    readme('01_current_figures/Figure_S2','Representative size and thickness cases. The saved table contains six branches of four plate models along the path. Reuse the closing Gamma value only for display.')
    meshes=['M1L4','M1L6','M2L2','M2L4','M2L6','M3L2','M3L4']
    for i,mesh in enumerate(meshes,1):
        copy('Figure S3',f'01_current_figures/Figure_S3/{mesh}.png',ASSETS+f'item-{i:03}.png','Saved COMSOL mesh image for the large circular cell; not a solver model or connectivity file.')
    readme('01_current_figures/Figure_S3','Saved mesh images and current assembled vector figure. Panels a-g are M3L2, M3L4, M2L2, M2L4, M2L6, M1L4, M1L6. The PNGs preserve the archived export bytes. These are illustrations, not mesh-connectivity data or .mph models.')
    # Convenient unrounded inputs are selected without carrying superseded analyses into current results.
    numeric_source='03_review/sensitivity_v26/NUMERIC_EVIDENCE.json';num=load(ROOT/numeric_source)
    selected_fields=['band','frequency_MHz','R_mc_matrix','S_ell','R_sh_spatial','kappa_minus1','kappa_1','kappa_3','eta_1','eta_3','S_h_analytic']
    cwrite('Table 1','02_current_tables/Table_1/unrounded_modal_response.csv',[{k:r[k] for k in selected_fields} for r in num['response_records']],numeric_source,'Saved full-precision numeric values used for the displayed response table; R_sh uses spatial reconstruction.')
    jwrite('Table 1','02_current_tables/Table_1/unrounded_gap_derivatives.json',dict(scope=num['scope'],gap_derivatives=num['gap_derivatives']),numeric_source,'Natural-log derivatives: relative gap in percentage points, absolute width in MHz.')
    cwrite('Table 2','02_current_tables/Table_2/unrounded_current_gains.csv',[r for r in csvrows('output/uniform_b024_v47/primary_table.csv') if r['model'] in ['PWE N15','3D M2L4']],'output/uniform_b024_v47/primary_table.csv','Only the four current rows; gap percent and gain percentage points. Historical optimized N21 rows are excluded.')
    for key,dest,source,note in [
        ('S1','unrounded_first_order_checks.csv','03_review/sensitivity_v26/plot_source.csv','16 comparisons, dimensionless log shifts and MAC.'),
        ('S2','unrounded_uniform_check.csv','06_release/github_v20/repository/reproducibility/wave_motion_20261015/uniform_recomputed/uniform_summary_recomputed.csv','Unrounded homogeneous periodic-solid recomputed checks; units follow headers.'),
        ('S3','unrounded_Rayleigh_Lamb_comparison.csv','05_validation/V04_Rayleigh_Lamb_20261030/Rayleigh_Lamb_A0_comparison.csv','Unrounded classical free-plate benchmark; units follow headers.'),
        ('S4','large_cell_mesh_errors.csv',ASSETS+'item-019.csv','Large-cell same-law band-edge error records.'),
        ('S4','endpoint_order_mesh_errors.csv',ASSETS+'item-012.csv','Contains both N15 current endpoints and N21 circular comparison checks; retain N column.'),
        ('S5','PWE_solid_MAC_sampling.csv',ASSETS+'item-013.csv','PWE-solid MAC at two sampling grids; does not supply all solid-solid source fields.'),
        ('S6','pointwise_comparison.csv','output/uniform_b024_v47/solid_point_comparison.csv','Unrounded optimized-interface pointwise frequency and field comparisons.'),
        ('S6','mesh_eigenpair_changes.csv','output/uniform_b024_v47/solid_mesh_count_changes.csv','Unrounded mesh and eigenpair-count frequency changes.')]:
        copy('Table '+key,'02_current_tables/Table_'+key+'/'+dest,source,note)
    readme('03_optimized_shapes','Five own-length-scale locally optimized designs, all N=15, b=0.24, 153 points. The units of vertices are x/a,y/a; spectra are dimensionless Omega. metadata.json gives cell dimensions and area fraction. summary.csv includes unrounded gaps, radius ratios, bounds and convergence quantities. The selected-source hashes preserve provenance. No historical mixed-bound or cross-applied geometry is included here.')
    # Complete manifest and privacy screening before any write.
    all_rows=[]
    for row in DISPLAY:
        row['export_sha256']=sha(FILES[row['path']])
        all_rows.append(row)
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(all_rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(all_rows)
    FILES['DISPLAY_MAP.csv']=stream.getvalue().encode('utf-8-sig')
    assert all(not re.search(rb'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY',data) for data in FILES.values())
    for dest,data in FILES.items():
        p=OUT/dest;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    report=dict(status='PASS_CURRENT_RESULT_EXPORT',manuscript_revision='v76',new_solves=0,
                figures=9,tables=9,optimized_shapes=5,files=len(FILES),bytes=sum(map(len,FILES.values())),
                copied_files=sum(r['export_mode']=='byte-identical copy' for r in DISPLAY),
                table_precision='Displayed CSVs retain the article rounding; separately supplied source tables retain saved precision.',
                unresolved_source_scope=['Table S5(a): all displayed values are transcribed; some unrounded solid-solid sampling source files were not located.'],
                source_hashes={r['source_path']:r['source_sha256'] for r in DISPLAY},
                file_sha256={p:sha(data) for p,data in sorted(FILES.items())},human_verified=False)
    REVIEW.mkdir(parents=True,exist_ok=True)
    (REVIEW/'CURRENT_EXPORT_QA.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['status','figures','tables','optimized_shapes','files','bytes','copied_files']}))


if __name__=='__main__':
    main()
