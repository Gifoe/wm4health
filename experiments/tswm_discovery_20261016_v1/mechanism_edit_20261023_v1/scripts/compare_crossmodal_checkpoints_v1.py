"""Check whether the diagnosed projection changed during actual native training."""
import pathlib,json,hashlib,argparse,datetime
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--step',type=int,choices=[500,1000,2000],required=True);args=p.parse_args();OUT=ROOT/'crossmodal_checkpoint_comparisons_v1';OUT.mkdir(exist_ok=True)
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
initial=read(ROOT/'factual_fit_v1/VALIDATION_STEP0000_V1.json');trained=read(ROOT/'factual_fit_recovery_v1'/f'VALIDATION_STEP{args.step:04d}_V1.json');assert sha(initial['checkpoint_path'])==initial['checkpoint_sha256'] and sha(trained['checkpoint_path'])==trained['checkpoint_sha256']
with np.load(initial['checkpoint_path'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
with np.load(trained['checkpoint_path'],allow_pickle=False) as z:b={k:z[k].copy() for k in z.files}
assert set(a)==set(b);d={k:float(np.max(abs(a[k].astype(np.float64)-b[k].astype(np.float64)))) for k in a};forward={k:v for k,v in d.items() if '/recon_wm/' in k};reverse={k:v for k,v in d.items() if '/recon_ctxc/' in k};decoder={k:v for k,v in d.items() if k.startswith('wm/dec/')}
assert forward and reverse and decoder;assert all(x==0 for x in forward.values());assert max(reverse.values())>0 and max(decoder.values())>0
result=dict(status='FORWARD_CROSSMODAL_PROJECTION_UNCHANGED_IN_ACTUAL_NATIVE_FIT',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),step=args.step,initial_checkpoint_sha256=initial['checkpoint_sha256'],trained_checkpoint_sha256=trained['checkpoint_sha256'],forward_projection_changes=forward,reverse_projection_changes=reverse,decoder_max_change=max(decoder.values()),other_parameters_changed=sum(x>0 for x in d.values()),prediction_failure_cause_established=False,paper_experiment_revision_known=False)
with (OUT/f'STEP{args.step:04d}_V1.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps(result),flush=True)
