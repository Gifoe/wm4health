"""Adapt native checkpoint field names; never change a numerical forward."""
from native53 import *
import torch

def load_compatible_cell(probe,model,arm,seed,checkpoint,bank,schema):
    checkpoint=pathlib.Path(checkpoint)
    original_path=probe.ckpt_path
    probe.ckpt_path=lambda *unused,**kwargs:checkpoint
    try:cell=probe.load_cell('greenhouse',model,arm,seed,0.,bank,schema)
    finally:probe.ckpt_path=original_path
    assert not isinstance(cell,str),cell
    blob=torch.load(checkpoint,map_location='cpu',weights_only=False)
    state=blob['state'];aliases=[]
    # RQ3's native concat trainer saves this branch as fusion_mlp, while
    # RQ2's concat probe loader expects fusion. Restore the saved branch after
    # the native constructor, without rewriting weights/checkpoint/source.
    if arm=='concat' and cell.adapter.CovariateFusion is not None and 'fusion_mlp' in state:
        cell.adapter.CovariateFusion.load_state_dict(state['fusion_mlp'])
        cell.adapter.CovariateFusion.eval();aliases.append('fusion_mlp -> concat covariate fusion')
    pairs=[('model',cell.adapter.model),('embedder',cell.embedder)]
    if 'target_ae' in state:pairs.append(('target_ae',cell.target_ae))
    if arm in ['gate','gate_soft'] and 'fusion' in state:pairs.append(('fusion',cell.fusion))
    if cell.adapter.CovariateFusion is not None:
        key='fusion_mlp' if 'fusion_mlp' in state else 'fusion'
        if key in state:pairs.append((key,cell.adapter.CovariateFusion))
    if cell.encoder is not None and 'encoder' in state:pairs.append(('encoder',cell.encoder))
    count=0
    for key,module in pairs:
        actual=module.state_dict();assert set(actual)==set(state[key]),key
        for name,tensor in actual.items():
            assert torch.equal(tensor.detach().cpu(),state[key][name]),(key,name)
            count+=1
    return cell,{'checkpoint_sha256':sha(checkpoint),'restored_state_tensors':count,'checkpoint_field_aliases':aliases,'original_checkpoint_rewritten':False}
