"""Finite empirical witness using an exact plant, not a learned baseline.

Can the unchanged native objective prefer a physically wrong perturbation over
the perfect known predictor? Select perturbation on train only; report on fresh
validation and test. No theorem or new method is asserted.
"""
import runtime_profile
import json,pathlib
import torch
from torch import nn
import native_action_probe as probe
ROOT=pathlib.Path(__file__).resolve().parents[1]
class OracleWorld(nn.Module):
 def __init__(self,kind,delta):
  super().__init__();self.kind=kind;self.delta=float(delta);self.action_encoder=nn.Identity();self.predictor=type('Monitor',(),{'transformer':type('Layers',(),{'layers':[]})()})()
 def encode(self,b):return {'emb':b['states'],'act_emb':b['action']}
 def predict(self,x,a):
  y=probe.plant(x[:,-1],a[:,-1],self.kind);y=y.clone();y[:,1]+=self.delta*a[:,-1,0];return y[:,None]
def main():
 torch.set_num_threads(1);probe.NATIVE=probe.load_native();out=ROOT/'outputs/objective_certificate';out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():raise RuntimeError('Existing certificate; do not overwrite')
 plan={'scope':'Finite held-out witness about source-default objective, exact known plant plus a spurious action force. Privileged objective diagnostic, not learned-method comparison or new theorem.','delta_grid':[-.6,.6,121],'seeds':[31,32,33],'cohort':'Fresh seeds61001/61002/61003; train4096,val2048,test4096, all-null dynamics,30% exact-zero action support','selection':'Minimum native training objective selects delta; test never selects it','test':'Compare physically exact delta0 with training-selected delta at unchanged source-default loss weights'}
 probe.dump(out/'plan.json',plan);train=probe.cohort(4096,'null',61001);val=probe.cohort(2048,'null',61002);test=probe.cohort(4096,'null',61003);rows=[]
 for seed in plan['seeds']:
  torch.manual_seed(seed);h=probe.Harness().cuda()
  for arm in ['hinge','full']:
   grid=[]
   with torch.no_grad():
    for delta in torch.linspace(-.6,.6,121).tolist():
     h.model=OracleWorld('null',delta);loss=h(train,arm);grid.append({'delta':delta,**{k:float(v) for k,v in loss.items() if k.endswith('loss')}})
    selected=min(grid,key=lambda x:x['loss']);cases=[]
    for delta in [0.,selected['delta']]:
     h.model=OracleWorld('null',delta);v=h(val,arm);t=h(test,arm)
     cases.append({'delta':delta,'validation_loss':float(v['loss']),'validation_factual_MSE':float(v['pred_loss']),'test_loss':float(t['loss']),'test_factual_MSE':float(t['pred_loss'])})
   row={'seed':seed,'arm':arm,'train_selected_delta':selected['delta'],'true_vs_perturbed':cases,'training_grid':grid};rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='training_grid'}),flush=True)
 probe.dump(out/'result.json',{'plan':plan,'rows':rows,'scope':'Default-loss objective witness only; no conclusion that all validation-tuned weights or all inverse-dynamics methods fail'})
if __name__=='__main__':main()
