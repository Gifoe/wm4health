"""One fixed-budget fit: training windows only, complete validation selection, no test scores."""
from common60 import *
from model60_v1 import *
import sys,time,numpy as np

def main():
    system=sys.argv[1];seed=int(sys.argv[2]);assert system in ['N87','pick_place'] and seed in [2021,2022,2023]
    source_check();qualified=read(OUT/'QUALIFICATION60_V1.json');assert qualified['status'].startswith('PASS_')
    assert qualified['model_adapter_sha256']==sha(ROOT/'scripts/model60_v1.py')
    device=qualified['chosen_device'];assert device=='cpu','The executed qualification chose CPU; additional CUDA numerical qualification is required before GPU fitting.'
    torch.set_num_threads(4);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    data=read(OUT/'DATA60_V2.json');entries=data['systems'][system]['splits'];arr={}
    for name in ['train','validation']:
        row=next(x for x in entries if x['split']==name);assert sha(row['file'])==row['sha256']
        with np.load(row['file']) as a:arr[name]={k:a[k] for k in ['u','y']}
    tu,ty=arr['train']['u'],arr['train']['y'];vu=torch.from_numpy(arr['validation']['u']).transpose(0,1).to(device);vy=torch.from_numpy(arr['validation']['y']).transpose(0,1).to(device)
    tag=f'{system}_seed{seed}';out=OUT/'runs'/tag;out.mkdir(parents=True,exist_ok=False);cp=paths()['checkpoint_root']/tag;cp.mkdir(parents=True,exist_ok=False)
    model=torch.jit.script(PostUpdateSSM()).to(device);opt=torch.optim.Adam(model.parameters(),lr=.0005)
    init=cp/'initial.pt';torch.save({'model':model.state_dict(),'seed':seed},init);initial_sha=sha(init)
    history=100;future=128;batchsize=32;steps=5000;best=None;started=time.perf_counter();curve=(out/'curve.jsonl').open('x',encoding='utf-8')
    def validate(step,train_loss=None):
        nonlocal best
        model.eval()
        with torch.no_grad():pred=model(vu[:history],vy[:history],vu[history:]);metric=float((pred-vy[history:]).square().mean())
        finite=bool(np.isfinite(metric));row={'utc':utc(),'step':step,'training_loss':train_loss,'validation_mse':metric if finite else None,'validation_finite':finite,'seconds':time.perf_counter()-started}
        if finite and (best is None or metric<best['validation_mse']):
            path=cp/f'selected_at_step{step:05d}.pt';torch.save({'model':model.state_dict(),'seed':seed,'step':step,'system':system,'protocol_sha256':sha(ROOT/'PROTOCOL60_V1.md')},path)
            best={'step':step,'validation_mse':metric,'checkpoint':str(path),'checkpoint_sha256':sha(path)};row['selected_new_best']=True
        curve.write(json.dumps(row,allow_nan=False)+'\n');curve.flush();print(json.dumps({'tag':tag,**row}),flush=True);model.train()
    validate(0)
    lastloss=None
    for step in range(1,steps+1):
        rows=rng.integers(0,len(tu),batchsize);starts=rng.integers(0,tu.shape[1]-history-future+1,batchsize);idx=starts[:,None]+np.arange(history+future)
        bu=torch.from_numpy(tu[rows[:,None],idx]).transpose(0,1).to(device);by=torch.from_numpy(ty[rows[:,None],idx]).transpose(0,1).to(device)
        opt.zero_grad();pred=model(bu[:history],by[:history],bu[history:]);loss=(pred-by[history:]).square().mean()
        if not torch.isfinite(loss):
            path=cp/f'failed_at_step{step:05d}.pt';torch.save({'model':model.state_dict(),'step':step},path);curve.close()
            dump(out/'result.json',{'utc':utc(),'status':'FAILED_NONFINITE_TRAINING_LOSS_PRESERVED','system':system,'seed':seed,'updates_completed':step-1,'failed_checkpoint':str(path),'checkpoint_sha256':sha(path),'heldout_scores_computed':False,'best_finite_validation':best});sys.exit(3)
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();lastloss=float(loss.detach())
        if step%250==0:validate(step,lastloss)
        elif step%100==0:
            row={'utc':utc(),'step':step,'training_loss':lastloss,'gradient_norm_before_clip':float(norm),'seconds':time.perf_counter()-started};curve.write(json.dumps(row)+'\n');curve.flush();print(json.dumps({'tag':tag,**row}),flush=True)
    curve.close();terminal=cp/'terminal_step05000.pt';torch.save({'model':model.state_dict(),'seed':seed,'step':steps,'system':system},terminal)
    assert best is not None
    obj={'utc':utc(),'status':'COMPLETE_FIXED_BUDGET_NSSM_VALIDATION_SELECTED','system':system,'seed':seed,'tag':tag,'updates':steps,'parameters':sum(p.numel() for p in model.parameters()),'device':device,'training_seconds':time.perf_counter()-started,'initial_checkpoint':str(init),'initial_checkpoint_sha256':initial_sha,'selected':best,'terminal_checkpoint':str(terminal),'terminal_checkpoint_sha256':sha(terminal),'final_training_loss':lastloss,'curve_sha256':sha(out/'curve.jsonl'),'data_manifest_sha256':sha(OUT/'DATA60_V2.json'),'qualification_sha256':sha(OUT/'QUALIFICATION60_V1.json'),'protocol_sha256':sha(ROOT/'PROTOCOL60_V1.md'),'script_sha256':sha(__file__),'model_adapter_sha256':sha(ROOT/'scripts/model60_v1.py'),'training_reads':['train','validation'],'test_score_selection':False,'source_commit':source_check()['source_commit'],'overall_goal':'ACTIVE_UNACHIEVED','accepted_final_directions':0}
    dump(out/'result.json',obj);print(json.dumps({'tag':tag,'status':obj['status'],'selected_step':best['step']}),flush=True)
if __name__=='__main__':main()
