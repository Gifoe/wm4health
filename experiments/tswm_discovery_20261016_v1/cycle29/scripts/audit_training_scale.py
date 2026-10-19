"""Train-only characterization of the fixed optimization objective; no test access."""
import runtime_profile
import json,pathlib,numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];DATA=ROOT/'outputs/world_data';OUT=ROOT/'outputs/world_verification'
def main():
    y=np.concatenate([np.load(DATA/f'train_{family}.npz',allow_pickle=False)['y'] for family in ['lti','wh']]);u=np.concatenate([np.load(DATA/f'train_{family}.npz',allow_pickle=False)['u'] for family in ['lti','wh']])
    rng=np.random.default_rng(2026291+1000);idx=rng.integers(0,len(y),size=(8192,32),dtype=np.int64);start=rng.integers(0,353,size=(8192,32),dtype=np.int64);energy=[];stds=[];ustds=[]
    for lo in range(0,8192,64):
        ids=idx[lo:lo+64].reshape(-1);s=start[lo:lo+64].reshape(-1);t=s[:,None]+np.arange(160)[None,:];yy=y[ids[:,None],t];uu=u[ids[:,None],t]
        mean=yy[:,:128].mean(1);std=yy[:,:128].std(1);scale=np.maximum(std,1e-3)
        energy.append(np.mean(((yy[:,128:]-mean[:,None])/scale[:,None])**2,1));stds.append(std);ustds.append(uu[:,:128].std(1))
    energy=np.concatenate(energy).astype('float64');std=np.concatenate(stds);us=np.concatenate(ustds);rank=np.sort(energy)[::-1]
    result=dict(status='completed',scope='train windows only, seed1 actual full schedule; persistence-at-context-mean target energy, not trained-model error',n_windows=len(energy),normalized_target_energy_mean=float(energy.mean()),target_energy_percentiles={str(q):float(np.quantile(energy,q)) for q in [.5,.9,.99,.999,1.]},y_context_std_percentiles={str(q):float(np.quantile(std,q)) for q in [0,.001,.01,.1,.5,1.]},u_context_std_percentiles={str(q):float(np.quantile(us,q)) for q in [0,.001,.01,.1,.5,1.]},fraction_y_std_below_001=float(np.mean(std<.01)),top_energy_fraction={str(q):float(rank[:max(1,int(len(rank)*q))].sum()/rank.sum()) for q in [.0001,.001,.01,.1]},no_test_read=True,no_protocol_change=True)
    OUT.mkdir(exist_ok=True);(OUT/'training_scale_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
