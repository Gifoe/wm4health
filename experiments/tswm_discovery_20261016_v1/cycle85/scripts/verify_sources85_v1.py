"""Replay every selected source checkpoint, including all unqualified ancestors."""
from common85_v1 import *
import verify85_v2 as replay

stages=[('primary',O,'anchored_screen_v1'),('capacity_v2',O/'capacity_v2','anchored_screen_capacity_v2'),('source_polish_v3',O/'source_polish_v3','anchored_screen_source_polish_v3')]
rows=[];original_data={};max_difference=0.
for stage,path,data_name in stages:
    result=json.loads((path/'RESULT85_V1.json').read_text())
    for record in result['sources']:
        assert sha(record['checkpoint'])==record['checkpoint_sha256']
        data_path=Path(CONFIG['data_root'])/'cycle85'/data_name/f"{record['family']}_{record['seed']}"/'source_trainval_v1.npz'
        assert sha(data_path)==record['source_data_sha256'];data=np.load(data_path,allow_pickle=False)
        key=(record['family'],record['seed'])
        if stage=='primary':original_data[key]={k:data[k] for k in data.files}
        else:assert all(np.array_equal(data[k],original_data[key][k]) for k in data.files)
        state=torch.load(record['checkpoint'],map_location='cpu',weights_only=True);agent=replay.model(record['nu'],state)
        assert sum(z.numel() for group in state.values() for z in group.values())==record['trained_parameters']
        predicted=predict(agent,data['val_y'][:,0],data['val_u']);m32=replay.metric(predicted,data['val_y'],32);m128=replay.metric(predicted,data['val_y'],128)
        checked=dict(mean32=float(np.mean(m32)),median32=float(np.median(m32)),median128=float(np.median(m128)))
        discrepancy=max(abs(v-record['selected'][k]) for k,v in checked.items());max_difference=max(max_difference,discrepancy)
        assert discrepancy<=1e-5
        assert record['qualified']==(checked['median32']<=.15 and checked['median128']<=.35)
        assert record['qualified']==any(v['median32']<=.15 and v['median128']<=.35 for v in record['validation_checks'])
        rows.append(dict(stage=stage,family=record['family'],seed=record['seed'],qualified=record['qualified'],source_data_sha256=record['source_data_sha256'],checkpoint_sha256=record['checkpoint_sha256'],parameter_count=record['trained_parameters'],maximum_score_difference=discrepancy,recomputed=checked))
result=dict(status='PASS',utc=utc(),source_checkpoints=len(rows),max_source_score_recompute_difference=max_difference,unchanged_gate_applied=True,source_arrays_identical_across_repair_stages=True,rows=rows,local_result_payload_files_written=0)
write(O/'SOURCE_VERIFICATION85_V1.json',result);print(json.dumps(result,indent=2),flush=True)
