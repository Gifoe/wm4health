"""CPU known-process EVAL references, with predeclared numerical escalation."""
import pathlib,sys,json,hashlib,math,datetime,os,time,traceback,ast
R=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(R.parent/'cycle29/scripts'));import runtime_profile
import numpy as np
H=8;FUT=16;PIXELS=64
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(p,v):pathlib.Path(p).write_text(json.dumps(v,indent=2),encoding='utf-8')
pilot=read(R/'outputs/pilot_v1/pilot_summary.json');assert pilot['status']=='PASS' and len(pilot['allocation_rows'])==240
validated=read(R/'outputs/reference_validation_v1/reference_validation.json');assert validated['status']=='REFERENCE_READY' and not validated['eval_opened']
# Use exactly the previously validated numerical helper bodies; no top-level replay.
kernel=R/'scripts/reference_validation44.py';tree=ast.parse(kernel.read_text(encoding='utf-8'));names={'evolve','projected_history','gaussian_posterior','particle_posterior','future_functions','mean_function_oracle'};helpers=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names];assert len(helpers)==len(names);exec(compile(ast.Module(body=helpers,type_ignores=[]),str(kernel),'exec'),globals())
O=R/'outputs/reference_eval_v1';assert not O.exists();O.mkdir(parents=True);state={'status':'running','pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'eval_opened':True,'contexts_completed':0};dump(O/'status.json',state)
def compare(left,right,scales):
 lm=np.mean([x['mean'] for x in left],0);rm=np.mean([x['mean'] for x in right],0);se=np.sqrt(sum(np.asarray(x['conditional_mc_se'])**2 for x in left+right))/2;delta=np.abs(lm-rm);limit=np.maximum(5*se,.02*scales);return {'left_particles':left[0]['particles'],'right_particles':right[0]['particles'],'scaled_difference':(delta/scales).tolist(),'scaled_tolerance':(limit/scales).tolist(),'pass_by_function':(delta<=limit).tolist()}
try:
 records=[];checks=[];truth_means=[]
 for case in ['gaussian','nonlinear']:
  data=R/'data'/case;eval_obs=np.load(data/'eval.npz')['observations'];truth=read(data/'truth_model.json');normalization=read(data/'normalization.json');spec=read(data/'functional_spec.json');scales=np.asarray(spec['scales'])
  for context in range(4):
   history=eval_obs[context,:H];local=[];exact=None
   if case=='gaussian':
    posterior=gaussian_posterior(history,truth,normalization);draw_seed=202645500+context;functions=future_functions(truth,normalization,spec,65536,draw_seed,posterior);raw=O/f'{case}_ctx{context}_exact.npz';np.savez_compressed(raw,functions=functions,posterior_mean=posterior[0],posterior_covariance=posterior[1]);mean=functions.mean(0);se=functions.std(0,ddof=1)/np.sqrt(len(functions));analytic_mean,analytic_variance=mean_function_oracle(*posterior,truth,normalization);exact={'case':case,'context':context,'kind':'exact_Kalman_conditional_MC','future_draws':len(functions),'future_seed':draw_seed,'mean':mean.tolist(),'conditional_mc_se':se.tolist(),'analytic_mean_function':analytic_mean,'analytic_mean_function_variance':analytic_variance,'raw_sha256':sha(raw)};records.append(exact)
   comparisons=[];previous=None
   for n in [16384,65536,262144]:
    if n==262144 and all(comparisons[-1]['pass_by_function']):break
    group=[]
    for rep in range(2):
     particle_seed=202645600+(case=='nonlinear')*100+context*10+rep;future_seed=particle_seed+10000;posterior=particle_posterior(history,truth,normalization,n,particle_seed);functions=future_functions(truth,normalization,spec,32768,future_seed,posterior);raw=O/f'{case}_ctx{context}_n{n}_r{rep}.npz';np.savez_compressed(raw,functions=functions);item={'case':case,'context':context,'kind':'bootstrap_particle_filter','particles':n,'rep':rep,'particle_seed':particle_seed,'future_seed':future_seed,'future_draws':len(functions),'history_ess':posterior[2],'mean':functions.mean(0).tolist(),'conditional_mc_se':(functions.std(0,ddof=1)/np.sqrt(len(functions))).tolist(),'raw_sha256':sha(raw)};group.append(item);local.append(item);records.append(item)
    if previous is not None:comparisons.append(compare(previous,group,scales))
    previous=group
   final=comparisons[-1];ready=all(final['pass_by_function']);gm=np.mean([x['mean'] for x in group],0);gs=np.sqrt(sum(np.asarray(x['conditional_mc_se'])**2 for x in group))/2
   check={'case':case,'context':context,'count_comparisons':comparisons,'selected_particle_count':group[0]['particles'],'count_convergence_pass':ready}
   if exact is not None:
    delta=np.abs(gm-exact['mean']);limit=np.maximum(5*np.sqrt(gs**2+np.asarray(exact['conditional_mc_se'])**2),.02*scales);check.update(PF_vs_exact_scaled_difference=(delta/scales).tolist(),PF_vs_exact_pass=(delta<=limit).tolist());ready=ready and bool(np.all(delta<=limit));estimate=np.asarray(exact['mean']);uncertainty=np.asarray(exact['conditional_mc_se']);kind=exact['kind']
   else:estimate=gm;uncertainty=gs;kind='bootstrap_particle_filter'
   check['ready']=ready;checks.append(check);truth_means.append({'case':case,'context':context,'kind':kind,'mean_scaled':(estimate/scales).tolist(),'conditional_mc_se_scaled':(uncertainty/scales).tolist(),'between_filter_sd_scaled':(np.std([x['mean'] for x in group],axis=0,ddof=1)/scales).tolist(),'ready':ready});state.update(contexts_completed=state['contexts_completed']+1,current_context=f'{case}_{context}');dump(O/'status.json',state);print(json.dumps(check),flush=True)
 result={'status':'REFERENCE_READY' if all(x['ready'] for x in checks) else 'REFERENCE_UNCERTAINTY_UNRESOLVED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'records':records,'checks':checks,'truth_estimates':truth_means,'eval_opened':True,'source_sha256':sha(pathlib.Path(__file__)),'qualified_kernel_sha256':sha(kernel),'evaluation_rule_sha256':sha(R/'EVALUATION_RULE.md'),'pilot_sha256':sha(R/'outputs/pilot_v1/pilot_summary.json'),'no_hidden_EVAL_states_used':True};dump(O/'reference_eval.json',result);state.update(status='completed',result_status=result['status'],completed_utc=result['utc']);dump(O/'status.json',state)
except Exception:
 state.update(status='failed',traceback=traceback.format_exc());dump(O/'status.json',state);raise
