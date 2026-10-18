from native62 import *
from cached_loader62_v1 import install
ad,models,compat=load_native();install();df,p=load_data();a=adapter(ad,models,'Informer_PRO');a.multi_forecasting_hyper_param_tune(df.iloc[:1000]);a.config.task_name='short_term_forecast';util=importlib.import_module('ts_benchmark.baselines.utils')
ds,dl=util.forecasting_data_provider(df.iloc[:1000],a.config,1,4,False,False);batch=next(iter(dl));xx,yy,xm,ym=batch;dec=torch.cat([yy[:,:48],torch.zeros_like(yy[:,-96:])],1)
seed(62902);pro=models['Informer_PRO'](a.config);dyn=models['Informer_DYN'](a.config);mapped={k.replace('linear_PRO','linear_DYN'):v.clone() for k,v in pro.state_dict().items()};dyn.load_state_dict(mapped)
rows=[]
for training in [False,True]:
 outputs=[];gradients=[]
 for model in [pro,dyn]:
  model.train(training);model.zero_grad(set_to_none=True);seed(62903);out=model(xx,xm,dec,ym);out.square().mean().backward();outputs.append(out.detach().clone());gradients.append({k.replace('linear_PRO','linear_DYN'):v.grad.detach().clone() for k,v in model.named_parameters() if v.grad is not None})
 forward=float((outputs[0]-outputs[1]).abs().max());grad=max(float((gradients[0][k]-gradients[1][k]).abs().max()) for k in gradients[0]);assert forward==0. and grad==0.,(training,forward,grad)
 rows.append({'training_mode':training,'forward_max_difference':forward,'gradient_max_difference':grad})
dump(OUT/'PRO_DYN_EQUIVALENCE62_V1.json',{'utc':utc(),'status':'EXACT_EQUIVALENCE_AT_L_EQUALS_H','seq_len':96,'pred_len':96,'mapped_parameter_names':'linear_PRO -> linear_DYN','state_dictionary_shapes_match':True,'cpu_checks':rows,'scope':'At L=H the PRO short_forecast else branch is the DYN computation. No claim of equivalence when L differs from H.','interpretation':'Not a mechanism-isolating three-architecture comparison. Two named variants are repeated executions of one architecture here.','optimizer_updates':0})
print(json.dumps({'status':'EXACT_EQUIVALENCE_AT_L_EQUALS_H','checks':rows}))
