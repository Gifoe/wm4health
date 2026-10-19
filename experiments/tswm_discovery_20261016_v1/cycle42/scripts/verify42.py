"""Independent NumPy conditioner + scaled HMM recursion, all 32 selected fits."""
import sys,pathlib,json,hashlib,itertools,datetime
R=pathlib.Path(__file__).resolve().parents[1];B=R.parent
sys.path.insert(0,str(B/'cycle29/scripts'));import runtime_profile
import numpy as np
import torch
O=R/'outputs/fits_v1';report=json.loads((O/'results.json').read_text());assert report['status']=='PASS'
cells=[(v,s) for v in range(4) for s in range(4)];seen=[(v,s) for v,s in cells if s in [v,(v+1)%4]]
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def independent_scores(data,a,c):
 out=np.zeros(data.shape[:2]);probs=np.empty((*data.shape[:2],4))
 for j,cell in enumerate(cells):
  aa=a[j];cc=c[j];y=data[j];belief=np.full((len(y),4),.25);scores=[]
  for t in range(y.shape[1]):
   if t:belief=belief@aa
   weighted=belief*cc[:,y[:,t]].T;z=weighted.sum(1);belief=weighted/z[:,None]
   if t>=8:scores.append(-np.log(z))
   if t==7:probs[j]=belief@aa@cc
  out[j]=np.mean(scores,axis=0)
 return out,probs
def numpy_matrix(weights,kind,om,dm):
 vv=np.eye(4)[np.asarray([v for v,s in cells])];ss=np.eye(4)[np.asarray([s for v,s in cells])]
 x=np.concatenate([vv,ss*0 if om else ss],1) if kind=='emission' else np.concatenate([vv*0 if dm else vv,ss],1)
 p=lambda k:weights[k].detach().cpu().numpy().astype(np.float64)
 hidden=np.tanh(x@p(kind+'.0.weight').T+p(kind+'.0.bias'))
 logits=(hidden@p(kind+'.2.weight').T+p(kind+'.2.bias')).reshape(16,4,4)+(1 if kind=='emission' else 2)*np.eye(4)
 logits-=logits.max(-1,keepdims=True);out=np.exp(logits);return out/out.sum(-1,keepdims=True)
maxnll=maxp=maxkl=maxmat=maxsummary=0.;count=0;metriccells=0
allraw={};oracles={}
for w in range(2):
 d=np.load(O/f'world{w}.npz');on,op=independent_scores(d['eval'],d['A'][[s for v,s in cells]],d['C'][[v for v,s in cells]])
 stored=np.load(O/f'oracle{w}.npz');assert np.max(np.abs(on-stored['nll']))<1e-11 and np.max(np.abs(op-stored['next_probability']))<1e-11
 oracles[w]=(on,op);allraw[w]=d['eval']
for folder in sorted(O.glob('w*_n*_i*_o*d*')):
 ckpt=torch.load(folder/'model.pt',map_location='cpu',weights_only=False);raw=np.load(folder/'scores.npz');meta=load(folder/'summary.json');row=meta[0];w=row['world'];om,dm=row['obs_mod'],row['dyn_mod']
 assert sha(folder/'model.pt')==row['checkpoint_sha256'] and sha(folder/'scores.npz')==row['score_sha256']
 a=numpy_matrix(ckpt['selected'],'transition',om,dm);c=numpy_matrix(ckpt['selected'],'emission',om,dm)
 maxmat=max(maxmat,float(np.abs(a-raw['learned_A']).max()),float(np.abs(c-raw['learned_C']).max()))
 nll,p=independent_scores(allraw[w],a,c);on,op=oracles[w];kl=np.sum(op*np.log(op/p),axis=-1)
 maxnll=max(maxnll,float(np.abs(nll-raw['nll']).max()));maxp=max(maxp,float(np.abs(p-raw['next_probability']).max()));maxkl=max(maxkl,float(np.abs(kl-raw['oracle_next_kl']).max()))
 curves=load(folder/'curves.json');assert ckpt['selected_step']==min(curves,key=lambda x:x['seen_val_future_nll'])['step']
 for result in meta:
  ids=[j for j,x in enumerate(cells) if (x in seen)==(result['group']=='seen')]
  independent={'future_nll':nll[ids].mean(),'oracle_future_nll':on[ids].mean(),'excess_nll':(nll[ids]-on[ids]).mean(),'next_symbol_oracle_kl':kl[ids].mean()}
  for key,value in independent.items():maxsummary=max(maxsummary,abs(float(value)-result[key]));metriccells+=1
 count+=1
assert count==32 and len(report['rows'])==64 and metriccells==256
# Float32 training scorers vs independent Float64 recursion, fixed tolerance.
assert max(maxnll,maxp,maxkl,maxmat,maxsummary)<2e-5,(maxnll,maxp,maxkl,maxmat,maxsummary)
v={'status':'PASS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'independently_rebuilt_models':count,'raw_sequence_scores':32*16*256,'next_distribution_cells':32*16*256*4,'summary_metric_cells':metriccells,'max_sequence_nll_abs':maxnll,'max_next_probability_abs':maxp,'max_next_kl_abs':maxkl,'max_matrix_abs':maxmat,'max_summary_abs':maxsummary,'tolerance':2e-5,'synthetic_evaluation_scored':True,'old_real_test_scored':False,'results_sha256':sha(O/'results.json'),'verifier_sha256':sha(pathlib.Path(__file__))}
(R/'outputs/verification_v1.json').write_text(json.dumps(v,indent=2),encoding='utf-8');print(json.dumps(v),flush=True)
