"""CPU changed-path equality before queueing deployment controls; no model scores."""
import runtime_profile
import copy,json
import torch
from world_models_v2 import build
from inference_implementations import prepare_fused_gru,fused_simulate,enable_unused_attention_weight_elision
from audit_inference_implementations import difference,cell_check
from world_data import ROOT,sha,dump

@torch.no_grad()
def main():
    torch.set_num_threads(1);torch.manual_seed(202629301)
    cy=torch.randn(1,128,1);cu=torch.randn_like(cy);fu=torch.randn(16,128,1)
    checks=[]
    for kind in ['conditional96','conditional128','compiled64','native_attention64']:
        model=build(kind).cpu().eval();encoded=model.encode(cy,cu)
        batch_encoded=tuple(v.expand((16,)+v.shape[1:]).contiguous() for v in encoded)
        ref=model.simulate(batch_encoded,fu)
        if kind=='native_attention64':
            other=copy.deepcopy(model);count=enable_unused_attention_weight_elision(other)
            other_encoded=other.encode(cy,cu);other_encoded=tuple(v.expand((16,)+v.shape[1:]).contiguous() for v in other_encoded)
            actual=other.simulate(other_encoded,fu);step=None
        else:
            actual=fused_simulate(model,prepare_fused_gru(model,encoded),fu)
            step=cell_check(model,encoded,fu[:1]);count=None
        checks.append(dict(kind=kind,equivalence=difference(ref,actual),one_step=step,attention_modules=count))
    result=dict(status='passed',checks=checks,scope='CPU randomized changed-path arithmetic only; no trained accuracy or latency result',script_sha256=sha(ROOT/'scripts/check_inference_implementation_math.py'),implementation_sha256=sha(ROOT/'scripts/inference_implementations.py'))
    dump(ROOT/'outputs/inference_implementation_math.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
