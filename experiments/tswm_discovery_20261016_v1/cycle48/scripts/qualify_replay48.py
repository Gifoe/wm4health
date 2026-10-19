"""One TRAIN-only critical-path check for the new independent NumPy replay."""
from common48_v3 import *
from independent_model48 import replay
def main():
 out=R/'outputs/replay_qualification_v1';assert not out.exists();out.mkdir();data=load('ett','train');history=data['history'][:2];phase=data['phase'][:2];records=[]
 for method in METHODS:
  seed(202648909);model=MemoryModel(method);head=model.core.linear if method=='gru' else next(x for x in reversed(list(model.core.mlp.children())) if isinstance(x,nn.Linear));nn.init.normal_(head.weight,std=.02);model.eval();before=checksum(model)
  with torch.no_grad():actual=model(history,phase).numpy()
  weights={k:v.detach().numpy() for k,v in model.state_dict().items()};independent=replay(method,weights,history.numpy(),phase.numpy(),48);error=float(np.max(abs(actual-independent)));assert np.allclose(actual,independent,atol=3e-4,rtol=3e-4),(method,error);np.savez_compressed(out/(method+'.npz'),native=actual,independent=independent,history=history.numpy(),phase=phase.numpy(),**weights);assert checksum(model)==before;records.append({'method':method,'max_absolute_forecast_error':error,'formal_weights_used':False,'optimization_updates':0})
 result={'status':'PASS_INDEPENDENT_NUMPY_REPLAY_TRAIN_ONLY','utc':now(),'checks':records,'EVAL_opened':False,'source_sha256':sha(__file__),'independent_source_sha256':sha(R/'scripts/independent_model48.py'),'common_sha256':sha(R/'scripts/common48_v3.py')};dump(out/'qualification.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
