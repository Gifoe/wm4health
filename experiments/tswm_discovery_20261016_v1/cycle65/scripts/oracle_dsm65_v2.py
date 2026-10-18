from models65 import *
import scipy.linalg as sla
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest');torch.set_num_threads(4)
registry=OUT/'FITS65_V1.json' if (OUT/'FITS65_V1.json').exists() else OUT/'fit_registry65_v1.json';f=read(registry);device=read(OUT/'QUALIFICATION65_V1.json')['device'];rows=[];assert sum(x['mechanism']=='linear' for x in f['corpora'])==3
a=.8-.7*.75;G=np.zeros((H,H))
for t in range(H):G[t,:t+1]=a**np.arange(t,-1,-1)
ym=a**(np.arange(H)+1)*.25;um=-.75*np.r_[.25,ym[:-1]];MY=np.c_[.5*G,.7*.5*G];MU=-.75*np.eye(H,k=-1)@MY+np.c_[np.zeros((H,H)),.5*np.eye(H)];M=np.r_[MY,MU];mu=np.r_[ym,um];C=M@M.T
model=csdi(device)
for corpus in f['corpora']:
 if corpus['mechanism']!='linear':continue
 assert sha(corpus['path'])==corpus['sha256'];z=np.load(corpus['path']);mean=z['train'].mean((0,2),keepdims=True);std=z['train'].std((0,2),keepdims=True);raw=(z['validation'][:256]-mean)/std;sd=np.repeat(std.reshape(-1),H);mm=np.repeat(mean.reshape(-1),H);muN=(mu-mm)/sd;CN=C/np.outer(sd,sd);x=torch.tensor(raw,device=device)
 for kind in ['block','prefix']:
  m=np.zeros((256,2,H));cuts=np.random.default_rng(650004).integers(0,H,size=256)
  if kind=='block':m[:,1]=1
  else:
   for j,t in enumerate(cuts):m[j,0,:t]=1;m[j,1,:t+1]=1
  flat=m.reshape(256,-1).astype(bool);patterns=np.unique(flat,axis=0);groups=[];denom=float((1-m).sum());vf=raw.reshape(256,-1).astype('float64')
  for pattern in patterns:
   ids=np.flatnonzero(np.all(flat==pattern,axis=1));O=np.flatnonzero(pattern);U=np.flatnonzero(~pattern);B=sla.solve(CN[np.ix_(O,O)],CN[np.ix_(O,U)],assume_a='pos').T;cm=muN[U]+(vf[np.ix_(ids,O)]-muN[O])@B.T;cc=CN[np.ix_(U,U)]-B@CN[np.ix_(O,U)];ee,V=sla.eigh((cc+cc.T)/2);assert ee.min()>0;groups.append((ids,U,cm,ee,V))
  actual=[];population=[]
  with torch.no_grad(),torch.random.fork_rng(devices=[0] if device=='cuda' else []):
   torch.manual_seed(650003)
   for t in range(50):
    noise=torch.randn_like(x);alpha=model.alpha_torch[torch.full((len(x),),t,dtype=torch.long,device=device)];noisy=(alpha.sqrt()*x+(1-alpha).sqrt()*noise).cpu().numpy().reshape(256,-1).astype('float64');eps=noise.cpu().numpy().reshape(256,-1).astype('float64');aa=float(model.alpha[t]);error=0.;risk=0.
    for ids,U,cm,ev,V in groups:
     invN=(V/(aa*ev+1-aa))@V.T;pred=np.sqrt(1-aa)*(noisy[np.ix_(ids,U)]-np.sqrt(aa)*cm)@invN;error+=float(np.sum((eps[np.ix_(ids,U)]-pred)**2));risk+=len(ids)*float(np.sum(1-(1-aa)/(aa*ev+1-aa)))
    actual.append(error/denom);population.append(risk/denom)
  rows.append(dict(key=corpus['key'],view=kind,realized_Bayes_validation_DSM=float(np.mean(actual)),population_irreducible_DSM=float(np.mean(population))))
dump(OUT/'ORACLE_DSM65_V1.json',dict(utc=utc(),status='LINEAR_VALIDATION_ORACLE_EVALUATED',actual_script='oracle_dsm65_v2.py',rows=rows,source_registry=str(registry),source_registry_sha256=sha(registry),data_only_reference=True,scope='Linear Gaussian conditional Bayes denoiser, fixed native validation noise/masks; no nonlinear oracle, learned performance or model selection change.',note_sha256=sha(ROOT/'ORACLE_RISK_NOTE65_V1.md')));print(json.dumps({'status':'LINEAR_VALIDATION_ORACLE_EVALUATED','rows':rows}))
