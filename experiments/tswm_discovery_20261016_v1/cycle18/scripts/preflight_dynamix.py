"""Source-native CPU interface smoke test; no benchmark accuracy inference."""
import runtime_profile
import hashlib,json,pathlib,random,sys,time
import numpy as np
import torch
from safetensors.torch import load_file
ROOT=pathlib.Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    started=time.perf_counter();torch.set_num_threads(1);torch.manual_seed(2021);np.random.seed(2021);random.seed(2021)
    assert sys.version_info>=(3,11)
    out=ROOT/'outputs/preflight';assert not out.exists();out.mkdir(parents=True)
    source=ROOT/'upstream/DynaMix-python';manifest=json.loads((ROOT/'outputs/source/DynaMix/manifest.json').read_text())
    for row in manifest['files']:assert sha(source/row['path'])==row['sha256']
    cfg_path=ROOT/'outputs/external/DynaMix/config_3d-alrnn-v1.0.json'
    weight=ROOT/'outputs/external/DynaMix/dynamix-3d-alrnn-v1.0.safetensors'
    assert sha(cfg_path)=='faf9e84a6a93b13d929e20ebdcf8963bab7a6dca1d0462e9513f98d74cadf261'
    assert sha(weight)=='c950aef03f24e600bb159a05bf64aaea0fc89a6c66310baac624ac4b7e10ed5b'
    sys.path.insert(0,str(source/'src'))
    from dynamix.model.dynamix import DynaMix
    from dynamix.model.forecaster import DynaMixForecaster
    cfg=json.loads(cfg_path.read_text())['architecture']
    model=DynaMix(**cfg).cpu().eval();model.load_state_dict(load_file(str(weight),device='cpu'),strict=True)
    forecaster=DynaMixForecaster(model)
    t=torch.arange(400,dtype=torch.float32)
    context=torch.stack([torch.sin(t*.1),torch.cos(t*.14),torch.sin(t*.23+.5)],1)
    def forecast(seed,horizon):
        torch.manual_seed(seed)
        return forecaster.forecast(context,horizon,preprocessing_method='zero_embedding',standardize=True,
            fit_nonstationary=False,initial_x=None)
    first=forecast(2021,32);replay=forecast(2021,32);prefix=forecast(2021,16);different=forecast(2022,32)
    assert first.shape==(32,3) and all(torch.isfinite(x).all() for x in [first,replay,prefix,different])
    replay_error=float((first-replay).abs().max());prefix_error=float((first[:16]-prefix).abs().max())
    assert replay_error==prefix_error==0
    np.savez_compressed(out/'arrays.npz',context=context.numpy(),forecast=first.numpy(),
        replay=replay.numpy(),short_prefix=prefix.numpy(),different_seed=different.numpy())
    for row in manifest['files']:assert sha(source/row['path'])==row['sha256']
    result={'strict_native_state_dict':True,'source_revision':manifest['revision'],'source_files_verified':len(manifest['files']),
        'parameters':sum(p.numel() for p in model.parameters()),'architecture':cfg,'model_sha256':sha(weight),
        'same_seed_replay_error':replay_error,'same_seed_horizon_prefix_error':prefix_error,
        'different_seed_max_difference':float((first-different).abs().max()),
        'learned_state_attention_noise':model.gating_network.sigma.detach().tolist(),
        'output_shape':list(first.shape),'environment':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'device':'cpu'},
        'script_sha256':sha(pathlib.Path(__file__)),'plan_sha256':sha(ROOT/'PREFLIGHT_PLAN.md'),
        'arrays_sha256':sha(out/'arrays.npz'),'elapsed_seconds':time.perf_counter()-started,
        'scope':'Synthetic-input native-interface smoke test; neither paper metric reproduction nor held-out-system accuracy.',
        'new_training':False,'accuracy_claim':False,'accepted_novel_directions':0,'objective_achieved':False}
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
