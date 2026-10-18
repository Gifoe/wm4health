from native87_v1 import *
from ordinary87_v1 import identify,predict
from train87_v1 import dataset,get_batch,validation
from fim.models.ode_trainer import ODEConcepts
def main():
 d=dataset('val','unselected');m=load_native();m.eval()
 for p in m.parameters():p.requires_grad_(False)
 data=get_batch(d,np.arange(len(d['context'])));original=validation(m,d);rows=[]
 with torch.no_grad():
  result=m.model_forward(data['obs_values'],data['obs_times'],data['locations'],data['obs_mask']);stats=result.predictions
  def concept(raw):return ODEConcepts(locations=data['locations'],drift=raw,normalized=False,states_norm=m.spatial_norm,states_norm_stats=stats.states_norm_stats,times_norm=m.temporal_norm,times_norm_stats=stats.times_norm_stats)
  target=concept(data['drift_at_locations']);target.normalize();result.predictions.normalize();native_replay=float(m._criterion(result.predictions.drift,target.drift,data['dimension_mask']).mean());assert abs(native_replay-original)<1e-3
  ordinary=read(O/'ORDINARY_SELECTION87_V1.json')
  for row in ordinary['rows']:
   coef=identify(d['context'],row['degree'],row['ridge']);est=predict(coef,d['locations'],row['degree']);c=concept(torch.as_tensor(pad(est),device='cuda'));c.normalize();value=float(m._criterion(c.drift,target.drift,data['dimension_mask']).mean());rows.append(dict(degree=row['degree'],ridge=row['ridge'],exact_native_normalized_field_l1=value,previous_proxy=row['val_normalized_field_l1']))
 best=min(rows,key=lambda z:(z['exact_native_normalized_field_l1'],z['degree'],z['ridge']));same=(best['degree'],best['ridge'])==(ordinary['selected']['degree'],ordinary['selected']['ridge'])
 receipt=dict(utc=utc(),status='VAL_NORMALIZATION_REPLAY_CHECKED',frozen_native_val_l1=original,replayed_native_val_l1=native_replay,ordinary_exact_native_rows=rows,selected_unchanged=same,selected=best,eval_opened=False,weights_sha256=WEIGHTS['sha256'])
 write(O/'VAL_QUALIFICATION87_V1.json',receipt);log(dict(stage='val_qualification',frozen_native_l1=original,ordinary_native_l1=best['exact_native_normalized_field_l1'],ordinary_selected_unchanged=same));assert same,'Ordinary proxy changed selection: use versioned pre-EVAL correction, never proceed silently'
if __name__=='__main__':main()
