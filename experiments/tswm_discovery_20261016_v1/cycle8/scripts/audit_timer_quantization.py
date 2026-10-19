"""Measured timer-grid evidence; inspected source is not a flashed-version proof."""
import runtime_profile
import datetime,json,pathlib
import numpy as np,pandas as pd
ROOT=pathlib.Path(__file__).resolve().parents[1]
def diagnose(values,res):
 x=np.asarray(values,float);positive=x[x>0];period=(30000000. if res==1 else 30000.)/positive;steps=[1,2,4,8,16,32,64,128,256,512];coverage={str(g):float(np.mean(np.abs(period/g-np.round(period/g))<2e-4)) for g in steps};qual=[g for g in steps if coverage[str(g)]>.995]
 unique,counts=np.unique(positive,return_counts=True);order=np.argsort(counts)[::-1][:8]
 return {'samples':len(x),'zero_or_negative':int((x<=0).sum()),'resolution_flag':int(res),'period_unit':'microseconds' if res==1 else 'milliseconds','largest_tested_grid_with_99_5_percent_coverage':max(qual) if qual else None,'grid_coverage':coverage,'unique_RPM':len(unique),'top8_atom_fraction':float(counts[order].sum()/len(positive)),'top8_atoms':[{'rpm':float(unique[i]),'count':int(counts[i]),'equivalent_period':float((30000000. if res==1 else 30000.)/unique[i])} for i in order],'RPM_quantiles':np.quantile(x,[0,.01,.5,.99,1]).tolist()}
def main():
 out=ROOT/'outputs/timer_quantization';out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():raise RuntimeError('Existing diagnostic; preserve it')
 plan={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Test measured quantization atoms and source-consistent float-timer precision. No claim of physical latent recovery or known flashed firmware version.','formula':'Inspected source casts now and last_tick separately to float32 before subtraction; single precision spacing grows as2^(floor(log2(t))-23). For millisecond clocks spacing exceeds1ms after2^24ms, approximately4.66hours.','diagnostic':'Invert emitted RPM using source constants30000(res0,milliseconds) or30000000(res1,microseconds). Report coverage of power-of-two period grids; this pattern alone does not uniquely identify hardware or timing origin.'}
 (out/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8');rows=[]
 for p in sorted((ROOT/'data/wt_walks_v1').rglob('regime_jumps_*.csv')):
  df=pd.read_csv(p)
  for flag,g in df.groupby('flag',sort=False):
   for side in ['in','out']:
    vals=g['res_'+side].unique();assert len(vals)==1;res=int(vals[0]);row={'recording':p.stem,'regime':int(flag),'side':side,**diagnose(g['rpm_'+side],res)};rows.append(row)
  print('Audited timer atoms',p.name,flush=True)
 control=[]
 for p in sorted((ROOT/'data/wt_test_v1').rglob('tach_resolution.csv')):
  df=pd.read_csv(p)
  for (res,load),g in df.groupby(['res_in','load_in']):control.append({'res_in':int(res),'load_in':float(load),**diagnose(g.rpm_in,int(res))})
 model=json.loads((ROOT/'outputs/strong_transition_controls/results.json').read_text());contrib=[]
 for r in model['rows']:
  if not r['recording'].startswith('regime_jumps_'):continue
  groups=r['arms']['HGB_no_clock']['regimes'];low={x['regime'] for x in rows if x['recording']==r['recording'] and x['resolution_flag']==0};all_sse=sum(g['rows']*g['one_step_normalized_MSE'] for g in groups);low_sse=sum(g['rows']*g['one_step_normalized_MSE'] for g in groups if g['regime'] in low)
  contrib.append({'recording':r['recording'],'low_resolution_blocks':len(low),'all_blocks':len(groups),'low_resolution_row_fraction':sum(g['rows'] for g in groups if g['regime'] in low)/sum(g['rows'] for g in groups),'low_resolution_share_of_HGB_no_clock_squared_error':low_sse/all_sse})
 # A source-consistent numeric example, not reconstruction of an actual tick.
 examples=[]
 for now in [1_000_000,20_000_000,70_000_000,100_000_000]:
  old=now-10;exact=now-old;bad=float(np.float32(now)-np.float32(old));examples.append({'now_ms':now,'true_period_ms':exact,'float_before_difference_ms':bad,'float_after_unsigned_difference_ms':float(np.float32(np.uint32(now)-np.uint32(old))),'float32_spacing_ms':float(np.spacing(np.float32(now)))})
 result={'plan':plan,'rows':rows,'tach_calibration':control,'error_contribution':contrib,'numeric_examples':examples,'judgment':'Observed quantization and inspected implementation are a concrete competing measurement explanation. Do not label coarse-sensor score drops as a discovered latent-dynamics gap. Exact board firmware/provenance and physical truth remain unverified.'};(out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({'error_contribution':contrib,'numeric_examples':examples}),flush=True)
if __name__=='__main__':main()
