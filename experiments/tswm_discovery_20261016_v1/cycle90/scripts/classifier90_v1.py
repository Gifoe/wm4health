from common90_v1 import *
import torch,numpy as np,time
import csdi_core89_v1 as core
from torch.nn import functional as F

def new_classifier():
    return torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(288,128),torch.nn.SiLU(),torch.nn.Linear(128,128),torch.nn.SiLU(),torch.nn.Linear(128,9))

@torch.no_grad()
def validation(model,x,y):
    model.eval();logits=torch.cat([model(b) for b in x.split(256)]);pred=logits.argmax(1)
    return dict(cross_entropy=float(F.cross_entropy(logits,y)),accuracy=float((pred==y).float().mean()),macro_accuracy=float(torch.stack([(pred[y==c]==c).float().mean() for c in range(9)]).mean()),per_class_accuracy=[float((pred[y==c]==c).float().mean()) for c in range(9)])

def main():
    started=time.perf_counter();assert not (O/'CLASSIFIER90_V1.json').exists();manifest=read(ROOT/'cycle89/outputs/CSDI_DATA89_V1.json');paths={r['split']:r for r in manifest['files']};splits={}
    permutation=np.random.default_rng(20269061).permutation(9)
    for sp in ['train','cal']:
        assert sha(paths[sp]['path'])==paths[sp]['sha256']
        with np.load(paths[sp]['path']) as f:splits[sp]=(torch.as_tensor(f['X'].copy(),device='cuda'),torch.as_tensor(permutation[f['world']],device='cuda'))
    x,y=splits['train'];calx,caly=splits['cal'];prior=torch.bincount(y,minlength=9).double().cpu().numpy()/len(y)
    torch.manual_seed(2026906);model=new_classifier().cuda();opt=torch.optim.Adam(model.parameters(),lr=.001);scheduler=torch.optim.lr_scheduler.MultiStepLR(opt,[7500,9000],gamma=.1)
    dest=Path(STORAGE['checkpoint_root'])/'cycle90/classifier_v1';dest.mkdir(parents=True,exist_ok=True);cp=dest/'selected.pt';assert not cp.exists();best=float('inf');trace=[]
    for step in range(10001):
        if step in [0,1000,3000,6000,10000]:
            v=validation(model,calx,caly);trace.append(dict(step=step,**v));log(dict(stage='mechanism_classifier_cal',step=step,**v))
            if v['cross_entropy']<best:best=v['cross_entropy'];torch.save(dict(model=model.state_dict(),step=step,permutation=permutation,train_prior=prior),cp)
        if step==10000:break
        model.train();ix=torch.randint(len(x),(256,),device='cuda');opt.zero_grad(set_to_none=True);loss=F.cross_entropy(model(x[ix]),y[ix]);assert torch.isfinite(loss);loss.backward();opt.step();scheduler.step()
    selected=torch.load(cp,map_location='cuda',weights_only=False);model.load_state_dict(selected['model']);v=validation(model,calx,caly)
    receipt=dict(utc=utc(),status='CAL_CLASSIFIER_PASS' if v['accuracy']>=.9 and v['macro_accuracy']>=.8 else 'CAL_CLASSIFIER_FAIL',selected_step=selected['step'],cal=v,trace=trace,checkpoint_path=str(cp),checkpoint_sha256=sha(cp),train_class_prior=prior.tolist(),class_permutation=permutation.tolist(),mechanism_ids_supervision_only=True,physical_parameter_values_used=False,eval_accessed=False,parameters=sum(p.numel() for p in model.parameters()),protocol_sha256=sha(C/'COUPLING_DIAGNOSTIC90_V1.md'),elapsed_seconds=time.perf_counter()-started)
    write(O/'CLASSIFIER90_V1.json',receipt);log(dict(stage='mechanism_classifier_closed',status=receipt['status'],cal=v,elapsed_seconds=receipt['elapsed_seconds']))

if __name__=='__main__':main()
