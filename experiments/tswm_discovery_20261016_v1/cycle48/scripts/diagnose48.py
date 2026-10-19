"""Frozen-fit VAL numerical and history-sensitivity diagnostics, no selection."""
from common48_v3 import *
def main():
 assert read(R/'outputs/supervisor_v1.json')['status']=='ALL_SIXTEEN_FITS_COMPLETE_EVAL_UNOPENED';out=R/'outputs/diagnostics_v1';assert not out.exists();out.mkdir();records=[];profiles=[]
 for case in CASES:
  val=load(case,'val');h=val['history'][:8];phase=val['phase'][:8];truth=val['future'][:8]
  for method in METHODS:
   for s in SEEDS:
    tag=f'{case}__{method}__seed{s}';folder=R/'outputs/fits'/tag;f=read(folder/'fit_summary.json');assert sha(folder/'selected.pt')==f['selected_sha256'];seed(s);model=MemoryModel(method);model.load_state_dict(torch.load(folder/'selected.pt',map_location='cpu',weights_only=True)['model']);model.eval();before=checksum(model)
    with torch.no_grad():
     coarse=model(h,phase);rms_error=float((coarse-truth).square().mean().sqrt());detail={}
     if method!='gru':
      half=model(h,phase,dt=DT/2);quarter=model(h,phase,dt=DT/4);rms_half=float((coarse-half).square().mean().sqrt());rms_quarter=float((coarse-quarter).square().mean().sqrt());detail={'coarse_half_RMS':rms_half,'coarse_quarter_RMS':rms_quarter,'half_quarter_RMS':float((half-quarter).square().mean().sqrt()),'coarse_quarter_over_prediction_RMSE':rms_quarter/max(rms_error,1e-12),'numerical_error_exceeds_5pct_forecast_RMSE':rms_quarter>.05*rms_error};np.savez_compressed(out/(tag+'__refinement.npz'),coarse=coarse.numpy(),half=half.numpy(),quarter=quarter.numpy(),truth=truth.numpy())
    hx=h.clone().requires_grad_(True);gradient=torch.autograd.grad(model(hx,phase).mean(),hx)[0].abs().mean(0).squeeze(-1).detach().numpy();profiles.append({'model':tag,'case':case,'method':method,'seed':s,'absolute_gradient_of_mean_future_by_history_position':gradient.tolist(),'functional':'mean of all48 future predictions; cancellation possible, not an information-loss or causal test'})
    records.append({'model':tag,'case':case,'method':method,'seed':s,'n_VAL_origins':len(h),'coarse_prediction_RMSE':rms_error,**detail,'delays_hours':(model.core.delays.detach().numpy()*24).tolist() if method in ['fixed_dde','learned_dde'] else []});assert before==checksum(model) and sha(folder/'selected.pt')==f['selected_sha256']
 dump(out/'numerical_refinement.json',records);dump(out/'history_profiles.json',profiles);dump(out/'manifest.json',{'utc':now(),'scope':'VAL-only postselection diagnostics; no EVAL or weight selection','source_sha256':sha(__file__),'files':[{'path':p.name,'sha256':sha(p)} for p in out.glob('*.npz')],'numerical_flags':sum(r.get('numerical_error_exceeds_5pct_forecast_RMSE',False) for r in records)});print(json.dumps({'status':'VAL_DIAGNOSTICS_COMPLETE','models':len(records),'numerical_flags':sum(r.get('numerical_error_exceeds_5pct_forecast_RMSE',False) for r in records)}),flush=True)
if __name__=='__main__':main()
