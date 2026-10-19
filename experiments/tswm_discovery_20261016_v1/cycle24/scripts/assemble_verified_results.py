import copy,json,pathlib,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/'outputs'/p).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
main=read('integration_repaired/results.json');ordinary=read('ordinary_filter_controls/results.json');ekf=read('ekf_density_repaired/results.json');particle=read('particle_moment_control/results.json');patch=read('full_tail_patches/results.json')
mv=read('integration_repaired/independent_verification.json');ov=read('ordinary_controls_verification.json');pv=read('full_tail_patches/independent_verification.json')
assert mv['status']==ov['status']==pv['status']=='pass'
assert mv['result_sha256']==sha(ROOT/'outputs/integration_repaired/results.json')
assert ov['filter_results_sha256']==sha(ROOT/'outputs/ordinary_filter_controls/results.json') and ov['particle_results_sha256']==sha(ROOT/'outputs/particle_moment_control/results.json')
assert pv['result_sha256']==sha(ROOT/'outputs/full_tail_patches/results.json')
for e in ekf['cells']:
    c=next(x for x in ordinary['cells'] if (x['kind'],x['seed'],x['n'],x['probe_u'])==(e['kind'],e['seed'],e['n'],e['probe_u']))
    c['actual_query']['ekf'].update(e['actual_query']);c['actual_query']['ekf']['grid_refinement_nll_max_error']=e['refined_nll_max_error'];c['actual_query']['ekf']['tail_integration_error']=e['tail35_to37_nll_max_error'];c['ekf_density_repaired']=True
for p in patch['patches']:
    common=(p['kind'],p['seed'],p['n'],p['probe_u']);arm=p['arm']
    if arm in ['native','no_update','reweighted']:
        c=next(x for x in main['cells'] if (x['kind'],x['checkpoint_seed'],x['context_size'],x['probe_u'])==common);target=c['actual_query'][arm]
    elif arm=='map_ggn_laplace':
        c=next(x for x in ordinary['cells'] if (x['kind'],x['seed'],x['n'],x['probe_u'])==common);target=c['actual_query'][arm]
    else:
        c=next(x for x in particle['cells'] if (x['kind'],x['seed'],x['n'],x['probe_u'])==common and x['importance_rep']==p['rep']);target=c['actual_query'][arm.removeprefix('particle')]
    target.update(p['actual_query']);target['all_context_tail_patch_indices']=p['flagged_contexts'];target['all_context_tail_patch_file']=p['patch_file']
bundle=dict(main=main,ordinary=ordinary,particle=particle,source_result_hashes={p:sha(ROOT/'outputs'/p) for p in ['integration_repaired/results.json','ordinary_filter_controls/results.json','ekf_density_repaired/results.json','particle_moment_control/results.json','full_tail_patches/results.json']},
    verification=dict(main=mv,ordinary=ov,tail_patches=pv),aggregation_note='All metrics from frozen arrays; native synthetic moments repaired, EKF density tail repaired, then sparse all-context tail patches. No old result overwritten.',accepted_novel_directions=0,objective_achieved=False)
OUT=ROOT/'outputs/verified_results';assert not OUT.exists();OUT.mkdir();f=OUT/'results.json';f.write_text(json.dumps(bundle,indent=2),encoding='utf-8');print(json.dumps(dict(path=str(f),sha256=sha(f),main_conditions=48,ordinary_conditions=48,particle_rows=144,accepted_directions=0)))
