import sys,pathlib,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common82_v1 import *
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial import Delaunay,cKDTree
PARAMS=[(.6,.15,1.1,.1),(.9,.08,-.8,.2),(.4,.25,.7,-.15),(1.1,.12,-1.,-.1)]
SEEDS=[2026821,2026822]
def target(x,p):
 x=np.asarray(x);a,b,c,d=p;return -a*x[...,0]-b*x[...,0]**3+c*x[...,1]+d*x[...,0]*x[...,1]
def system(y,p,k):return np.array([target(y,p),-.4*y[1]-.1*y[1]**3+k*y[2],-.3*y[2]+.05*y[0]])
def main():
 art=storage('data')/'controlled_pairs_v1';art.mkdir(exist_ok=False);rows=[];tg=np.linspace(0,8,256);started=time.perf_counter()
 dump(OUT/'DATA_OPENING82_V1.json',dict(utc=utc(),protocol_sha256=sha(ROOT/'PROTOCOL82_V1.md'),generator_sha256=sha(__file__),parameters=PARAMS,seeds=SEEDS,score_values_opened=False))
 for mid,p in enumerate(PARAMS):
  for seed in SEEDS:
   initial=np.random.default_rng(seed).normal(size=(8,3))*np.array([1,1,1.2]);contexts=[];diagnostics=[]
   for k in [0.,1.2]:
    paths=[];solvers=[]
    for y0 in initial:
     sol=solve_ivp(lambda t,y:system(y,p,k),(0,8),y0,t_eval=tg,method='DOP853',rtol=1e-9,atol=1e-11);assert sol.success and sol.y.shape==(3,256) and np.isfinite(sol.y).all();paths.append(sol.y.T);solvers.append(dict(success=bool(sol.success),nfev=sol.nfev))
    contexts.append(np.stack(paths));diagnostics.append(solvers)
   cloud=[x[:,:,:2].reshape(-1,2) for x in contexts];pooled=np.concatenate(cloud);m=pooled.mean(0);s=pooled.std(0);assert (s>1e-8).all();hulls=[Delaunay(x) for x in cloud];trees=[cKDTree((x-m)/s) for x in cloud];lo=np.maximum(cloud[0].min(0),cloud[1].min(0));hi=np.minimum(cloud[0].max(0),cloud[1].max(0));assert (hi>lo).all();rng=np.random.default_rng(seed+mid*10000+82);candidates=rng.uniform(lo,hi,size=(100000,2));inside=(hulls[0].find_simplex(candidates)>=0)&(hulls[1].find_simplex(candidates)>=0);distance=np.maximum(trees[0].query((candidates-m)/s)[0],trees[1].query((candidates-m)/s)[0]);accepted=candidates[inside&(distance<=.1)];assert len(accepted)>=256,('Frozen common-coverage qualification failed',mid,seed,len(accepted));queries=accepted[:256];truth=target(queries,p)
   for index,k in enumerate([0.,1.2]):
    path=art/f'm{mid}_seed{seed}_k{index}.npz';np.savez_compressed(path,context=contexts[index],time=tg,queries=queries,field_target=truth,initial_states=initial,coverage_mean=m,coverage_std=s,query_distance=distance[inside&(distance<=.1)][:256]);rows.append(dict(mechanism=mid,seed=seed,k=k,regime='closed' if k==0 else 'open',parameters=p,path=str(path),sha256=sha(path),accepted_common_candidates=len(accepted),context_solver_diagnostics=diagnostics[index],query_count=256))
 dump(OUT/'DATA_MANIFEST82_V1.json',dict(utc=utc(),protocol_sha256=sha(ROOT/'PROTOCOL82_V1.md'),rows=rows,context_tasks=len(rows),clean_measurements=True,wall_seconds=time.perf_counter()-started));print(json.dumps(dict(status='DATA_FROZEN',tasks=len(rows),manifest_sha256=sha(OUT/'DATA_MANIFEST82_V1.json'))),flush=True)
if __name__=='__main__':main()
