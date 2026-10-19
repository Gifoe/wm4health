"""Existing predictive-moment dimension reduction control; no novel method claim."""
import datetime,hashlib,json,time
import numpy as np
import torch
from run_risk_state_pilot import ROOT,DEVICE,D,STEPS,Model,generate,evaluate,dump
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 parent=ROOT/'outputs'/'risk_state_pilot';assert json.loads((parent/'status.json').read_text())['state']=='complete'
 out=ROOT/'outputs'/'risk_moment_control';out.mkdir(parents=True,exist_ok=True)
 plan={'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Classical training cross-moment projection plus the same fitted prediction heads; additional existing-method control, not a method or original-paper reproduction','motivation':'Detached model consistently loses variance-driver information, but attached encoder repair is initialization-sensitive. Rule out a need for a new representation method using ordinary second predictive moments.','training_only':True,'uses_hidden_states_to_fit':False,'steps':STEPS,'seeds':[41,42,43],'data_seeds':{'train':10051,'probe':10052,'development':10053},'target_direction':'Reuse identical training OLS-derived output projection from the frozen pilot','encoder':'First row proportional to OLS current-observation mean coefficient. Second row from Cxx^-1 Cov(x, action*(r-OLS_mean)^2), orthogonalized against first. No true observation rotation used to fit either row. Standardize rows on training x, then freeze.','heads_and_updates':'Identical to matched pilot; no learning rate or test-set selection changes. 64 encoder coefficients frozen, same4485 trainable head parameters. Additional moment computation is reported, not hidden as exactly equal compute.','decision':'If ordinary moment projection retains risk state and reduces downstream regret, do not claim a new architecture is necessary. No candidate acceptance from this diagnostic.'}
 dump(out/'plan.json',plan)
 q=torch.tensor(np.load(parent/'observation_rotation.npy'),device=DEVICE)
 direction=torch.tensor(np.load(parent/'training_target_direction.npy'),device=DEVICE)
 train=generate(q,16384,10051);probe=generate(q,4096,10052);dev=generate(q,16384,10053)
 x=train['x'].double();a=train['action'].double();target=(train['future']@direction).double()
 design=torch.cat([x,a,torch.ones(len(x),1,device=DEVICE,dtype=torch.double)],1)
 beta=torch.linalg.lstsq(design,target).solution;residual=target-design@beta
 w1=beta[:D,0];w1=w1/w1.norm()
 xc=x-x.mean(0);score=a*residual.square();score=score-score.mean(0)
 w2=torch.linalg.solve(xc.T@xc/len(x),xc.T@score[:,0]/len(x));w2=w2-w1*(w1@w2);w2=w2/w2.norm()
 w=torch.stack([w1,w2]);std=(x@w.T).std(0);w=(w/std[:,None]).float()
 np.save(out/'training_only_encoder.npy',w.cpu().numpy())
 rows=[]
 for seed in plan['seeds']:
  run=out/f'seed{seed}';run.mkdir(exist_ok=True)
  if (run/'result.json').exists():rows.append(json.loads((run/'result.json').read_text()));continue
  torch.manual_seed(seed);model=Model('moment_projection').to(DEVICE)
  model.enc.weight.data.copy_(w);model.enc.weight.requires_grad_(False)
  opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=.0001)
  sampler=torch.Generator(device=DEVICE);sampler.manual_seed(10061);start=time.perf_counter()
  with (run/'train.jsonl').open('w',encoding='utf-8') as log:
   for step in range(1,STEPS+1):
    idx=torch.randint(0,16384,(256,),generator=sampler,device=DEVICE);r=train['future'][idx]@direction
    mu,lv,z=model(train['x'][idx],train['action'][idx]);mse=(mu-r).square().mean();nll=.5*(lv+(r-mu.detach()).square()/lv.exp()).mean();loss=mse+.1*nll
    opt.zero_grad(set_to_none=True);loss.backward();assert torch.isfinite(loss);opt.step()
    if step==1 or step%1000==0:
     log.write(json.dumps({'step':step,'loss':loss.item(),'mse':mse.item(),'nll':nll.item(),'seconds':time.perf_counter()-start})+'\n');log.flush()
     dump(out/'status.json',{'state':'training','seed':seed,'step':step,'completed':len(rows),'total':3})
  torch.save(model.state_dict(),run/'model.pt')
  row={'seed':seed,'mode':'training_cross_moment_projection','parameters_total':sum(p.numel() for p in model.parameters()),'parameters_trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),'seconds':time.perf_counter()-start,'checkpoint_sha256':hashlib.sha256((run/'model.pt').read_bytes()).hexdigest(),**evaluate(model,probe,dev,direction,q)}
  dump(run/'result.json',row);rows.append(row);dump(out/'results.json',rows);print(json.dumps(row),flush=True)
 dump(out/'status.json',{'state':'complete','completed':len(rows),'total':3,'candidate_accepted':False})
if __name__=='__main__':main()
