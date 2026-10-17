"""One fixed terminal fit; training opens train inputs/actions only."""
from common57 import *
import argparse,math,time,traceback
import numpy as np
import torch
from math57 import auxiliary,posterior,kernels,proper_loss,update_running
from models57 import Neural,Proper,describe
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False

def tensor_digest(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        a=v.detach().cpu().contiguous().numpy()
        h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('sensor',choices=SENSORS);parser.add_argument('method',choices=METHODS);parser.add_argument('seed',type=int,choices=SEEDS);args=parser.parse_args()
    assert args.method!='proper' or args.seed==2021
    qual=read(OUT/'QUALIFICATION57_V1.json');assert qual['status']=='PASS_NATIVE_LOSS_MOMENT_GRADIENT_CUDA_AND_PROPER_LIKELIHOOD'
    assert qual['math_sha256']==sha(ROOT/'scripts/math57.py') and qual['model_sha256']==sha(ROOT/'scripts/models57.py')
    assert qual['protocol_sha256']==sha(ROOT/'PROTOCOL.md');pins=source_check()
    plan=read(OUT/'TRAINING_PLAN57_V1.json');assert plan['train_script_sha256']==sha(__file__)
    tag=f'{args.method}_{args.sensor}_seed{args.seed}'
    run=OUT/'runs'/tag;run.mkdir(parents=True,exist_ok=False)
    storage=paths();ck=storage['checkpoint_root'];ck.mkdir(parents=True,exist_ok=True)
    data_path=storage['data_root']/'train_inputs_v1.npz'
    with np.load(data_path) as archive:
        # Physical state labels and all validation/heldout files are separate and never opened.
        xx=archive[args.sensor];aa=archive['actions']
    torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    dtype=torch.float64 if args.method=='proper' else torch.float32
    x=torch.tensor(xx,device='cuda',dtype=dtype);act=torch.tensor(aa,device='cuda',dtype=torch.long)
    model=(Proper() if args.method=='proper' else Neural()).to(device='cuda',dtype=dtype)
    initial_digest=tensor_digest(model.state_dict())
    initpath=ck/(tag+'_initial.pt');assert not initpath.exists()
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},initpath)
    rng=np.random.default_rng(57000+args.seed)
    memory=torch.full(x.shape[:2],-torch.inf,device='cuda',dtype=dtype) if args.method.startswith('running') else None
    last_visit=np.full(len(x),-1,dtype=np.int64)
    optimizer=(torch.optim.LBFGS(model.parameters(),lr=1,max_iter=200,max_eval=250,tolerance_grad=1e-7,tolerance_change=1e-9,history_size=30,line_search_fn='strong_wolfe') if args.method=='proper' else torch.optim.Adam(model.parameters(),lr=LR))
    curve=run/'curve.jsonl';torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter();evaluations=0
    def emit(row):
        row.update({'elapsed_seconds':time.perf_counter()-started,'model':describe(model)})
        with curve.open('a',encoding='utf-8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps({'tag':tag,**row}),flush=True)
    if args.method=='proper':
        def closure():
            nonlocal evaluations
            optimizer.zero_grad(set_to_none=True)
            loss=proper_loss(model.theta,model.lam,model.k,x,act)
            assert torch.isfinite(loss).item(),'Nonfinite proper likelihood'
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
            evaluations+=1
            if evaluations==1 or evaluations%10==0:emit({'likelihood_evaluation':evaluations,'train_nll':float(loss.detach())})
            return loss
        optimizer.step(closure);updates=int(optimizer.state[model.theta].get('n_iter',0))
        emit({'terminal_iterations':updates,'likelihood_evaluations':evaluations,'train_nll':float(proper_loss(model.theta,model.lam,model.k,x,act).detach()),'sensor_k':model.k.detach().cpu().tolist()})
    else:
        gamma=.9 if args.method=='running09' else .1
        for iteration in range(1,STEPS+1):
            idx=rng.choice(len(x),BATCH,replace=False);refidx=rng.choice(len(x),BATCH,replace=False)
            target=x[idx];actions=act[idx];means=model.encode(target)
            reference=means if args.method=='native' else model.encode(x if args.method=='full' else x[refidx])
            optimizer.zero_grad(set_to_none=True)
            post=posterior(model.theta,model.lam,means,actions)
            G,pos,qf=kernels(model.lam,means,reference,post)
            logmean=torch.logsumexp(G,-1)-math.log(reference.shape[0])
            if memory is None:denominator=logmean
            else:
                new=update_running(memory[idx],logmean,gamma)
                memory[idx]=new.detach();last_visit[idx]=iteration
                weights=torch.exp(G-new[...,None]).detach()
                denominator=(weights*G).mean(-1)
            loss=(-pos+denominator+qf).mean()+post['kl'].mean()/target.shape[1]
            assert torch.isfinite(loss).item(),f'Nonfinite objective at {iteration}'
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
            optimizer.step()
            if iteration==1 or iteration%100==0:
                emit({'iteration':iteration,'training_value':float(loss.detach()),'value_is_running_gradient_surrogate':memory is not None})
        updates=STEPS;evaluations=STEPS
    torch.cuda.synchronize();training_seconds=time.perf_counter()-started
    peak=torch.cuda.max_memory_allocated();common_objective=[];normalizer_errors=[]
    with torch.no_grad():
        ur=model.encode(x)
        for first in range(0,len(x),64):
            chunk=x[first:first+64];u=model.encode(chunk);acts=act[first:first+64]
            if args.method=='proper':
                value=proper_loss(model.theta,model.lam,model.k,chunk,acts)
            else:
                value,_,logmean=auxiliary(model.theta,model.lam,u,ur,acts)
                if memory is not None:normalizer_errors.append((memory[first:first+len(chunk)]-logmean).cpu().numpy())
            common_objective.append((len(chunk),float(value)))
    assert all(torch.isfinite(v).all().item() for v in model.state_dict().values())
    checkpoint=ck/(tag+'.pt');assert not checkpoint.exists()
    payload={'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':optimizer.state_dict(),'method':args.method,'sensor':args.sensor,'seed':args.seed,'dtype':str(dtype),'updates':updates,'batch_rng_state':rng.bit_generator.state,'torch_rng_state':torch.get_rng_state(),'cuda_rng_state':torch.cuda.get_rng_state_all(),'normalizer_memory':None if memory is None else memory.cpu(),'normalizer_last_visit':last_visit}
    torch.save(payload,checkpoint)
    tracking=None
    if normalizer_errors:
        err=np.concatenate(normalizer_errors)
        assert np.isfinite(err).all()
        tracking={'log_normalizer_bias':float(err.mean()),'log_normalizer_rmse':float(np.sqrt(np.mean(err**2))),'absolute_error_percentiles':np.percentile(abs(err),[50,90,99]).tolist(),'mean_updates_since_visit':float((STEPS-last_visit).mean()),'interpretation':'Terminal finite-train-reference tracking error; entries are correlated, no iid interval.'}
    assert source_check()==pins
    result={'utc':utc(),'status':'COMPLETE_FIXED_TERMINAL_FIT','tag':tag,'method':args.method,'sensor':args.sensor,'seed':args.seed,'stochastic_fit':args.method!='proper','updates':updates,'objective_evaluations':evaluations,'training_seconds':training_seconds,'total_seconds':time.perf_counter()-started,'peak_training_cuda_bytes':peak,'dtype':str(dtype),'description':describe(model),'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'initial_checkpoint_sha256':sha(initpath),'initial_tensor_sha256':initial_digest,'terminal_tensor_sha256':tensor_digest(model.state_dict()),'train_inputs_sha256':sha(data_path),'protocol_sha256':sha(ROOT/'PROTOCOL.md'),'clarification_sha256':sha(ROOT/'EXECUTION_CLARIFICATION_V1.md'),'plan_sha256':sha(OUT/'TRAINING_PLAN57_V1.json'),'qualification_sha256':sha(OUT/'QUALIFICATION57_V1.json'),'train_script_sha256':sha(__file__),'math_sha256':sha(ROOT/'scripts/math57.py'),'model_sha256':sha(ROOT/'scripts/models57.py'),'common_train_objective':sum(n*v for n,v in common_objective)/len(x),'common_objective_type':'proper_negative_log_likelihood_per_time' if args.method=='proper' else 'full_512_reference_centered_auxiliary_per_time','normalizer_tracking':tracking,'training_reads':['train_inputs_v1.npz'],'hidden_state_training_reads':0,'validation_training_reads':0,'heldout_training_reads':0,'checkpoint_selection':'fixed terminal, no score-based selection','target_sequences_per_neural_update':BATCH if args.method!='proper' else None,'encoded_sequences_per_neural_update':None if args.method=='proper' else BATCH+(0 if args.method=='native' else len(x) if args.method=='full' else BATCH)}
    dump(run/'result.json',result)
    print(json.dumps({'status':result['status'],'tag':tag,'updates':updates,'training_seconds':training_seconds,'peak_cuda_bytes':peak}),flush=True)

if __name__=='__main__':main()
