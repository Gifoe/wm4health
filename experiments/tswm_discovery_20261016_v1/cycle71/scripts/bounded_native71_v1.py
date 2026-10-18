"""Exact native epoch bodies with process-local affinity and bounded lifetimes."""
import sys,pathlib,argparse,ast,copy,types,hashlib,random,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from runtime_affinity71_v1 import configure
AFFINITY=configure('efficient')
from resume_native71_v2 import *

def bind_bounded(bg,resume):
    path=SOURCE/'modular_metalearning.py';tree=ast.parse(path.read_bytes())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BounceGrad')
    fn=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='SAConfig_SGDModules'))
    loop=next(n for n in fn.body if isinstance(n,ast.For))
    before=hashlib.sha256(ast.dump(ast.Module(body=loop.body,type_ignores=[]),include_attributes=False).encode()).hexdigest()
    assert before=='a983b27e66dcc1c9844a0333875f10473c02db44d7e67bd4397d3a0610e7f996'
    changes=[]
    if resume:
        for n in fn.body:
            if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='temp':
                n.value=ast.parse('self.resume_temperature71',mode='eval').body;changes.append('restore exact temperature')
            if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='self.S.initialize_all_structures':
                n.value=ast.parse('self.restore_continuation71()',mode='eval').body;changes.append('initialize bookkeeping then restore full checkpoint/RNG')
        assert len(changes)==2
    r=loop.iter.args[0];assert ast.unparse(r)=='range(optimization_steps)'
    r.args=[ast.parse('self.resume_start71' if resume else '0',mode='eval').body,
            ast.parse('min(self.boundary_stop71, optimization_steps)',mode='eval').body]
    changes.append('bound only the outer range; retain original optimization_steps')
    fn.body.extend(ast.parse('self.boundary_temperature71 = temp').body)
    assert hashlib.sha256(ast.dump(ast.Module(body=loop.body,type_ignores=[]),include_attributes=False).encode()).hexdigest()==before
    namespace=dict(BounceGrad.SAConfig_SGDModules.__globals__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(path)+'::bounded_continuation','exec'),namespace)
    bg.SAConfig_SGDModules=types.MethodType(namespace['SAConfig_SGDModules'],bg)
    return dict(original_source_sha256=sha(path),unchanged_epoch_loop_body_sha256=before,
                entry_only_changes=changes,exit_instrumentation='copy exact local temperature; no RNG or optimizer operation',
                transformed_function_ast_sha256=hashlib.sha256(ast.dump(fn,include_attributes=False).encode()).hexdigest())

def run(request):
    qualification=bool(request['qualification']);seed=int(request['seed']);total=int(request['total_epochs'])
    ntrain=int(request['train_tasks']);stop=int(request['stop']);tag=request['tag']
    assert tag.replace('_','').isalnum() and 0<stop<=total
    if qualification:assert seed==2026731 and total==3 and ntrain==128
    else:
        assert seed in (2026711,2026712,2026713) and total==2000 and ntrain==2048
        assert stop==2000 or (stop-1)%50==0
        assert read(OUT/'BOUNDED_CONTINUATION_QUALIFICATION71_V1.json')['status']=='PASS_BOUNDED_PROCESS_CONTINUATION_EXACT_EQUIVALENCE'
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.cuda.reset_peak_memory_stats()
    args=parse_native(seed,total,ntrain,tag);S=HGNN_Structure(args)
    manifest_path=request.get('parent_manifest')
    if manifest_path:
        manifest=read(manifest_path);record=manifest['checkpoints'][-1]
        assert record['sha256']==request['parent_checkpoint_sha256'] and sha(record['path'])==record['sha256']
        state=torch.load(record['path'],map_location='cpu',weights_only=False);start=int(state['epoch'])+1
        assert state['seed']==seed and start<stop and stop-start<=50
        bg=ResumedNative(S,args,tag)
        bg.configure_restore71(state,record,manifest['checkpoints'],float.fromhex(request['temperature_float64_hex']))
    else:
        assert not qualification and seed in (2026712,2026713)
        start=0;record=None;bg=InstrumentedNative(S,args,tag)
        assert stop==51
    bg.boundary_stop71=stop;adapter=bind_bounded(bg,bool(manifest_path))
    if manifest_path:bg.resume_adapter71=adapter
    assert bg.optimization_steps==total and not bg.auto_temp and bg.stop_edge_acc<=0
    bg.SAConfig_SGDModules(total)
    assert bg.step==stop-1
    expected=stop*((ntrain+127)//128)
    assert bg.module_updates71==bg.proposal_updates71==expected
    if not bg.receipts71 or bg.receipts71[-1]['epoch']!=bg.step:
        assert qualification,'Formal segment must end on the unchanged native checkpoint grid'
        bg.save_train_state()
    bg.writer.close()
    result=dict(utc=utc(),status='NATIVE_BOUNDED_SEGMENT_COMPLETE',seed=seed,tag=tag,
        qualification=qualification,performance_evidence=False if qualification else None,
        start_epoch=start,last_epoch=int(bg.step),total_epochs=total,train_tasks=ntrain,val_tasks=len(bg.T.MVAL),
        effective_module_updates=bg.module_updates71,effective_proposal_updates=bg.proposal_updates71,
        parent_checkpoint=record,checkpoint_manifest=str(OUT/(tag+'_checkpoints.json')),
        last=bg.receipts71[-1],selected=bg.selected71,
        temperature_float64_hex=float(bg.boundary_temperature71).hex(),
        temperature=float(bg.boundary_temperature71),entry_adapter=adapter,affinity=AFFINITY,
        module_parameters=sum(p.numel() for p in bg.L.parameters()),proposal_parameters=sum(p.numel() for p in bg.encoder.parameters()),
        wall_seconds=time.perf_counter()-bg.start71,gpu_peak_bytes=int(torch.cuda.max_memory_allocated()),
        reserved_eval_rows_accessed=0,source_bodies=EXECUTED_BODIES,
        protocol_sha256=sha(ROOT/'BOUNDED_RUNTIME_PROTOCOL71_V1.md'),root_cause_verified=False,
        long_run_stability_verified=False,accepted_final_directions=0)
    dump(OUT/(tag+'_segment_result.json'),result)
    print(json.dumps({k:result[k] for k in ['status','seed','tag','start_epoch','last_epoch','effective_module_updates','qualification','wall_seconds']}),flush=True)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--request',required=True);args=parser.parse_args()
    p=pathlib.Path(args.request).resolve();assert p.parent==OUT.resolve()
    run(read(p))
