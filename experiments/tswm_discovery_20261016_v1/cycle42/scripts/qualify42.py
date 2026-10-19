"""Independent small-law checks and runtime-only device choice; discard weights."""
from common42 import *
O=R/'outputs/qualification_v1';assert not O.exists();O.mkdir(parents=True)
seed(202642);m=Model(False,False).double();ids=torch.tensor([0,1]);sid=torch.tensor([1,2]);A,C=m.parameters_for(ids,sid)
y=torch.tensor([[0,2,1,3],[2,1,2,0]])
parts,_=forward_prob(y,A,C,need_at=2);errors=[]
for b in range(2):
 total=0.
 for states in itertools.product(range(4),repeat=4):
  p=.25*float(C[b,states[0],int(y[b,0])])
  for t in range(1,4):p*=float(A[b,states[t-1],states[t]]*C[b,states[t],int(y[b,t])])
  total+=p
 errors.append(abs(math.log(total)-float(parts[b].sum())))
assert max(errors)<1e-12
anchors=torch.tensor([[0,1,2,3],[3,2,0,1]])
_,ca=m.parameters_for(torch.arange(4),torch.zeros(4,dtype=torch.int64));manual=[]
for row in anchors:
 total=sum(.25*math.prod(float(ca[v,z,int(row[v])]) for v in range(4)) for z in range(4));manual.append(-math.log(total))
ae=abs(float(anchor_nll(anchors,m))-np.mean(manual));assert ae<1e-12
loss=-parts.sum();loss.backward();p=m.transition[-1].bias;idx=3;grad=float(p.grad[idx]);old=float(p.data[idx]);vals=[]
for sign in [1,-1]:
 p.data[idx]=old+sign*1e-5;a,c=m.parameters_for(ids,sid);vals.append(float(-forward_prob(y,a,c,2)[0].sum()))
p.data[idx]=old;graderr=abs((vals[0]-vals[1])/2e-5-grad);assert graderr<1e-7
# Prefix posterior is invariant to future corruption under both architectures.
for o,d in [(0,0),(0,1),(1,0),(1,1)]:
 test=Model(o,d).double();a,c=test.parameters_for(ids,sid);yy=torch.randint(0,4,(2,T));changed=yy.clone();changed[:,PREFIX:]=(changed[:,PREFIX:]+1)%4
 p1=forward_prob(yy,a,c)[1];p2=forward_prob(changed,a,c)[1];assert torch.equal(p1,p2)
 for q in [a,c]:assert torch.allclose(q.sum(-1),torch.ones_like(q.sum(-1)),atol=1e-12)
timings={}
for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
 seed(202642);model=Model(False,False).to(device);opt=torch.optim.Adam(model.parameters(),lr=.01)
 yy=torch.randint(0,4,(32,T),device=device);v=torch.arange(32,device=device)%4;s=(v+1)%4;anc=torch.randint(0,4,(32,4),device=device)
 for _ in range(5):opt.zero_grad();train_batch(model,yy,v,s,anc,128).backward();opt.step()
 if device=='cuda':torch.cuda.synchronize()
 start=time.perf_counter()
 for _ in range(30):opt.zero_grad();train_batch(model,yy,v,s,anc,128).backward();opt.step()
 if device=='cuda':torch.cuda.synchronize()
 timings[device]=(time.perf_counter()-start)/30
chosen=min(timings,key=timings.get)
result={'status':'PASS','brute_force_likelihood_max_abs':max(errors),'independent_anchor_abs':ae,'finite_difference_gradient_abs':graderr,'runtime_seconds_per_update':timings,'chosen_device':chosen,'training_per_device':35,'timing_weights_discarded':True,'protocol_sha256':sha(R/'PROTOCOL.md'),'source_sha256':sha(pathlib.Path(__file__)),'common_sha256':sha(R/'scripts/common42.py')}
dump(O/'qualification.json',result);print(json.dumps(result),flush=True)
