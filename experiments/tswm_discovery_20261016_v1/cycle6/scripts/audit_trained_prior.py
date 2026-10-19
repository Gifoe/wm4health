"""Full-real-line scalar conditional density checks at trained native priors."""
import runtime_profile
import datetime,hashlib,json,pathlib
import numpy as np,torch
from native_child import ROOT,build
from piecewise_density import audit
def main():
 torch.set_num_threads(1);run=ROOT/'outputs/child_native_A_r1'
 assert json.loads((run/'status.json').read_text())['state']=='complete','Wait for full native training to finish'
 out=ROOT/'outputs/trained_prior_audit';out.mkdir(parents=True,exist_ok=True)
 with np.load(ROOT/'data/native_A/A/data.npz') as data:
  ids=np.load(run/'split_indices.npz')['train'][[0,100,1000]];x=torch.tensor(data['xt'][ids],device='cuda',dtype=torch.float32)
 rows=[]
 for which in ['initial','best','final']:
  model=build(seed=770)
  if which!='initial':model.load_state_dict(torch.load(run/f'{which}.pt',map_location='cuda',weights_only=True))
  model.eval()
  with torch.no_grad():mus=model.forward({'xt':x,'yt':None})[2].cpu().double().numpy()
  # Only fixed training trajectories are used. This audit cannot select models.
  for i in range(7):
   layer=0 if i<1 else 1 if i<3 else 2;begin=[0,1,3][layer];end=[1,3,7][layer];parent_begin=[0,0,1][layer];parent_end=[0,1,3][layer]
   net=model.transition_prior_fix.gs[i].cpu().double()
   for case in range(3):
    parents=np.r_[mus[case,0,begin:end],mus[case,1,parent_begin:parent_end]]
    row={'checkpoint':which,'coordinate':i,'training_trajectory_id':int(ids[case]),'parents':parents.tolist(),**audit(net,parents)};rows.append(row)
  print(json.dumps({'checkpoint':which,'scalar_conditionals':21,'max_absolute_mass_error':max(r['absolute_mass_error'] for r in rows if r['checkpoint']==which),'min_base_reachable_mass':min(r['base_noise_mass_with_preimage'] for r in rows if r['checkpoint']==which)}),flush=True)
 result={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':rows,'scope':'Exploratory diagnostics on3fixed training contexts x7 scalar conditional factors xinitial/best/final; no global high-dimensional density certification, test selection or new-method claim. Initial comparator seed770 is a matched initialization seed, not claimed tensor-identical to the training initialization after split RNG consumption.'}
 (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
if __name__=='__main__':main()
