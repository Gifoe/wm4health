"""A train-independent mathematical diagnostic of the external-bank control."""
from common57 import *
os.environ['JAX_PLATFORM_NAME']='cpu';os.environ['JAX_ENABLE_X64']='true'
import numpy as np
import jax.numpy as j
sys.path.insert(0,str(PROJECT/'cycle56/scripts'))
from math56 import sample_loss

def main():
    # a=0 gives three independent unit-prior time points, not the native T=1 chain.
    x=j.ones((1,3));reference=j.zeros((4,3));self_reference=j.concatenate([x,reference[:3]],0)
    records=[]
    for jitter in [0.,1e-5]:
        for a in [0.,.1]:
            for b in [1.,2.,4.,8.,16.]:
                theta=j.array([a,0.,np.log(b)])
                ext=float(sample_loss(theta,x,reference,1.3,'external',False,jitter))
                included=float(sample_loss(theta,x,self_reference,1.3,'self',False,jitter))
                exact=.5*np.log(4/3)-b*b/12 if jitter==0 and a==0 else None
                if exact is not None:assert abs(ext-exact)<1e-10
                assert included>=-np.log(4)-1e-10
                records.append({'a':a,'lambda':1.,'b':b,'jitter':jitter,'external_loss':ext,'positive_included_loss':included,'closed_form_external_loss':exact})
    assert not (OUT/'HELDOUT_OPEN_RECEIPT57_V1.json').exists()
    result={'utc':utc(),'status':'PASS_EXTERNAL_BANK_SUPPORT_COUNTEREXAMPLE','records':records,'closed_form':'.5*log(4/3)-b^2/12 at a=0,lambda=1,ideal jitter','scope':'Per-batch mathematical counterexample for the adapted external-reference loss. Not original SogCLR execution, stochastic training evidence, a new method, or a universal claim about expected contrastive losses.','previous_native_matched_math_sha256':sha(PROJECT/'cycle56/scripts/math56.py'),'script_sha256':sha(__file__),'protocol_sha256':sha(ROOT/'PROTOCOL.md'),'source_files_sha256':source_check(),'formal_training_updates':0,'heldout_opened':False}
    dump(OUT/'INDEPENDENT_SUPPORT57_V1.json',result)
    text='''# Why independent references are a diagnostic control

The C57 independent-reference adaptation develops rapidly growing recognition precision in completed train-only runs. It is not a generally safe substitute for the native normalizer. This note tests the mathematical objective without reading any formal model, state label, validation or heldout observation.

Take a stationary scalar prior with a=0, three independent time points, unit site precision, recognition mean b*x, target x=(1,1,1), and four external reference sequences equal to zero. With ideal covariance arithmetic, the per-time centered loss is exactly

    L_external(b) = 0.5*log(4/3) - b^2/12.

The target posterior is N(b/2,1/2). The positive-minus-external log kernel contributes b^2/4; local KL contributes (0.5-log(1.5))/2+b^2/24, and prior KL contributes (-0.5+log(2))/2+b^2/8. Their combination yields the expression above. The loss is unbounded below as b grows even at fixed precision. The target lies outside that finite empirical reference support. Including the positive in a four-item bank instead guarantees the log-reference term is at least -log(4), and the KL terms are nonnegative.

The script verifies the exact expression and retains the literal reporting-jitter and nonzero-a cases separately. The ideal a=0 formula is not silently asserted for those cases. The independent bank here is a mathematical adaptation, not a native option or an execution of a bounded-similarity image-learning pipeline. The counterexample alone does not prove that every population expectation is unbounded. It explains why a failed external-bank control cannot establish that competent existing global-normalizer methods are insufficient.

No model or training rule was changed, no failed attempt was removed, and no heldout score was opened. All planned controls and seed outcomes remain visible. Centered running estimates and complete-reference controls, rather than this unsafe diagnostic alone, are required to assess the candidate repair.
'''
    with (ROOT/'INDEPENDENT_REFERENCE_CAVEAT_V1.md').open('x',encoding='utf-8') as f:f.write(text)
    print(json.dumps({'status':result['status'],'exact_max_error':max(abs(x['external_loss']-x['closed_form_external_loss']) for x in records if x['closed_form_external_loss'] is not None),'cases':len(records),'updates':0}),flush=True)

if __name__=='__main__':main()
