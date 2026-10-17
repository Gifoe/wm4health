"""Retain and describe the failed unshifted adaptation; never repair its result."""
from common57 import *
import torch,numpy as np

def main():
    assert read(OUT/'training_state57_v1.json')['status']=='FAILED_PRESERVED'
    assert read(OUT/'post_state57_v1.json')['status']=='FAILED_PRESERVED'
    assert not (OUT/'HELDOUT_OPEN_RECEIPT57_V1.json').exists()
    tag='running09_linear_seed2021';run=OUT/'runs'/tag;checkpoint=paths()['checkpoint_root']/(tag+'.pt')
    malformed=run/'result.json'
    try:read(malformed)
    except json.JSONDecodeError:pass
    else:raise AssertionError('Expected retained malformed result was changed')
    curve=[json.loads(s) for s in (run/'curve.jsonl').read_text().splitlines()]
    assert curve[-1]['iteration']==10000
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert payload['method']=='running09' and payload['updates']==10000
    assert all(torch.isfinite(v).all().item() for v in payload['model'].values())
    files=[malformed,run/'curve.jsonl',OUT/(tag+'.stdout.log'),OUT/(tag+'.stderr.log'),checkpoint,OUT/'training_state57_v1.json',OUT/'post_state57_v1.json']
    record={'utc':utc(),'status':'FAILED_UNSHIFTED_RUNNING_ADAPTATION_SEALED_EXCLUDED','tag':tag,'attempted_updates':10000,'terminal_parameter_tensors_finite':True,'terminal_common_reference_diagnostic':'inf; JSON serialization rejected it','last_curve_point':curve[-1],'retained_files':[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files],'valid_previous_fits':read(OUT/'training_state57_v1.json')['completed'],'heldout_open_receipt_absent':True,'validation_opened':False,'source_files_sha256':source_check(),'interpretation':'Direct unbounded-Gaussian-score adaptation failed numerically; not evidence that the original bounded-similarity SogCLR pipeline or all existing normalizer repairs fail.'}
    dump(OUT/'FAILED_ATTEMPT57_V1.json',record)
    print(json.dumps({'status':record['status'],'attempted_updates':10000,'valid_fits_retained':len(record['valid_previous_fits'])}),flush=True)

if __name__=='__main__':main()
