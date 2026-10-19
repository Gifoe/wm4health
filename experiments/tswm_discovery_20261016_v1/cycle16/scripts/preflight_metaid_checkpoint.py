"""Restricted native-checkpoint interface check; synthetic inputs, no accuracy claim."""
import runtime_profile
import argparse,hashlib,json,pathlib,sys,time
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1];SOURCE=ROOT/'upstream/sysid_benchmarks'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 start=time.perf_counter();torch.set_num_threads(1);torch.manual_seed(2021)
 out=ROOT/'outputs/metaid_preflight';assert not out.exists();out.mkdir(parents=True)
 p=ROOT/'outputs/external/metaid/ckpt_430_fine-tuned_3100000_lower_noise.pt'
 expected='e8324becad85ac610ec94f8ae09048938b343761a5bcd3967a86353a5bf28197'
 assert p.stat().st_size==90193824 and sha(p)==expected
 unsafe=set(torch.serialization.get_unsafe_globals_in_checkpoint(p));assert unsafe<={'argparse.Namespace'},unsafe
 # Namespace is a standard-library data container; no external class or pickle callable is enabled.
 with torch.serialization.safe_globals([argparse.Namespace]):payload=torch.load(p,map_location='cpu',weights_only=True)
 assert isinstance(payload,dict) and isinstance(payload['cfg'],argparse.Namespace) and isinstance(payload['model_args'],dict)
 manifest=json.loads((ROOT/'outputs/source/metaid/manifest.json').read_text())
 for row in manifest['files']:assert sha(SOURCE/row['path'])==row['sha256']
 sys.path.insert(0,str(SOURCE))
 from transformer_sim import Config,TSTransformer
 model= TSTransformer(Config(**payload['model_args'])).cpu();model.load_state_dict(payload['model'],strict=True);model.eval()
 cfg=payload['cfg'];ctx=cfg.seq_len_ctx;nin=cfg.seq_len_n_in;H=cfg.seq_len_new
 y=torch.randn(2,ctx,cfg.ny);u=torch.randn(2,ctx,cfg.nu);query_u=torch.randn(2,nin+H,cfg.nu)
 query_y=torch.zeros(2,nin+H,cfg.ny);query_y[:,:nin]=torch.randn(2,nin,cfg.ny)
 with torch.inference_mode():
  mean,std=model(y,u,query_u,query_y,nin)[:2]
  assert mean.shape==std.shape==(2,H,cfg.ny) and torch.isfinite(mean).all() and torch.isfinite(std).all() and (std>0).all()
  changed_y=query_y.clone();changed_y[:,nin:]=torch.randn_like(changed_y[:,nin:])*17
  alt=model(y,u,query_u,changed_y,nin)[:2];placeholder=[float((a-b).abs().max()) for a,b in zip([mean,std],alt)]
  assert max(placeholder)==0
  split=H//2;changed_u=query_u.clone();changed_u[:,nin+split:]=torch.randn_like(changed_u[:,nin+split:])*7
  alt=model(y,u,changed_u,query_y,nin)[:2]
  causal=[float((a[:,:split]-b[:,:split]).abs().max()) for a,b in zip([mean,std],alt)];assert max(causal)<2e-6
 safe_cfg={key:getattr(cfg,key,None) for key in ['seed','nx','nu','ny','seq_len_ctx','seq_len_skip','seq_len_n_in','seq_len_new',
             'seq_len_patch','d_model_RNN','n_layer','n_head','n_embd','dropout','bias','mag_range','phase_range']}
 result={'checkpoint_sha256':expected,'bytes':p.stat().st_size,'source_revision':manifest['revision'],
         'restricted_weights_only_load':True,'explicit_safe_extra_globals':sorted(unsafe),'source_files_verified':len(manifest['files']),
         'strict_state_dict_load':True,'parameters':sum(v.numel() for v in model.parameters()),'native_model_args':payload['model_args'],
         'native_training_config_subset':safe_cfg,'future_placeholder_errors':placeholder,'future_control_prefix_errors':causal,
         'output_shape':list(mean.shape),'finite_positive_standard_deviation':True,'elapsed_seconds':time.perf_counter()-start,
         'environment':{'python':sys.version,'torch':torch.__version__,'device':'cpu'},'script_sha256':sha(pathlib.Path(__file__)),
         'scope':'Synthetic-input interface preflight of released native model; not original metric reproduction or scientific accuracy experiment',
         'new_training':False,'benchmark_accuracy_claim':False,'new_method_claim':False,'objective_achieved':False}
 (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
