from gaussian64 import *
s=read(OUT/'SAMPLER_EVALUATION64_V1.json');rows=[];raw={};groups=[];errors=[]
for r in s['particle_rows']:
 if r['method']!='terminal_IS':continue
 k,h,P,seed=(r[q] for q in ['gain','horizon','particles','MC_seed']);rng=np.random.default_rng(np.random.SeedSequence([64200,seed,int(k*100),h,P]));u=plan(h,'sine');x=np.empty((P,h));last=np.full(P,Y0)
 for t in range(h):last=RHO*last+BETA*u[t]+SIGMA*rng.normal(size=P);x[:,t]=last
 m=x.mean(0);v=x-m;C=v.T@v/P;g=canonical(u,-k);o=metrics(m,C,g['do_mean'],g['do_cov']);o['Gaussian_moment_projection_W2_squared']=o.pop('gaussian_W2_squared');o['Gaussian_moment_tail_probability']=o['terminal_tail_probability'];o['terminal_tail_probability']=float(np.mean(np.abs(x[:,-1])>2));o['terminal_tail_abs_error']=abs(o['terminal_tail_probability']-o['true_terminal_tail_probability']);o.update(method='exact_target_iid_privileged_reference',gain=k,horizon=h,particles=P,MC_seed=seed);o['operational_pass']=o['normalized_Gaussian_moment_discrepancy']<=.1 and o['terminal_tail_abs_error']<=.03;rows.append(o);raw[f'iid_k{k}_H{h}_P{P}_s{seed}']=x
 # Independent Bures metric and direct tail replay from the retained draws.
 root=la.cholesky(g['do_cov'],lower=True);L=la.cholesky(C,lower=True);w2=max(0.,float((m-g['do_mean'])@(m-g['do_mean'])+np.trace(C)+np.trace(g['do_cov'])-2*la.svdvals(root.T@L).sum()));errors.append(abs(w2-o['Gaussian_moment_projection_W2_squared']));assert errors[-1]<1e-9
for k in [.75,2.]:
 for h in [16,64]:
  for P in [128,512]:
   rr=[r for r in rows if r['gain']==k and r['horizon']==h and r['particles']==P];ss=[r for r in s['particle_rows'] if r['method']!='terminal_IS' and r['gain']==k and r['horizon']==h and r['particles']==P];groups.append(dict(gain=k,horizon=h,particles=P,iid_passed=sum(r['operational_pass'] for r in rr),SMC_passed=sum(r['operational_pass'] for r in ss),replications=3,iid_mean_discrepancy=float(np.mean([r['normalized_Gaussian_moment_discrepancy'] for r in rr])),SMC_mean_discrepancy=float(np.mean([r['normalized_Gaussian_moment_discrepancy'] for r in ss])),iid_mean_tail_error=float(np.mean([r['terminal_tail_abs_error'] for r in rr])),SMC_mean_tail_error=float(np.mean([r['terminal_tail_abs_error'] for r in ss]))))
p=paths()['artifact_root']/'IID_REFERENCE64_V1.npz';np.savez_compressed(p,**raw);dump(OUT/'IID_REFERENCE64_V1.json',dict(utc=utc(),status='PASS',rows=rows,groups=groups,artifact=str(p),artifact_sha256=sha(p),independent_metric_max_absolute_error=max(errors),note_sha256=sha(ROOT/'IID_REFERENCE_NOTE64_V1.md'),scope='Post-screen privileged finite-particle floor, not deployable inference; no tuning or changed original settings'))
print(json.dumps({'rows':len(rows),'groups':groups,'independent_metric_max_absolute_error':max(errors)}))
