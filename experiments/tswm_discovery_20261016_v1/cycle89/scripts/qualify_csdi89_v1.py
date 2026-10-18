from csdi_core89_v1 import *
import time,platform

def main():
    started=time.perf_counter();assert not (O/'CSDI_QUALIFICATION89_V1.json').exists()
    model,meta=new_model(20268913)
    data=read(O/'CSDI_DATA89_V1.json');train=next(r for r in data['files'] if r['split']=='train')
    with np.load(train['path']) as f:x=torch.as_tensor(f['X'][:8].copy(),device='cuda')
    cm,tp=mask_and_time(len(x),'cuda');side=model.get_side_info(tp,cm)
    model.train();loss=model.calc_loss(x,cm,torch.ones_like(cm),side,1)
    assert torch.isfinite(loss);loss.backward()
    grad=[p.grad for p in model.parameters() if p.grad is not None]
    assert len(grad)>0 and all(torch.isfinite(g).all() for g in grad)
    assert any((g!=0).any() for g in grad)
    model.eval();original=x.detach().cpu().numpy();changed=original.copy();changed[:,:2,CONTEXT:]=17.
    first=sample(model,original,20268914);second=sample(model,changed,20268914)
    error=float(abs(first-second).max());assert error==0.
    receipt=dict(utc=utc(),status='NATIVE_CORE_FORWARD_BACKWARD_SAMPLING_QUALIFIED',source=meta,config=config(),parameters=sum(p.numel() for p in model.parameters()),finite_gradient_tensors=len(grad),loss=float(loss.detach()),sample_shape=list(first.shape),masked_future_dummy_change_max_abs=error,runtime=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0)),peak_allocated_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.perf_counter()-started,protocol_sha256=sha(C/'CSDI_COUPLING_PROTOCOL89_V1.md'),source_files_unchanged=all(sha(Path(meta['root'])/p)==h for p,h in meta['sha256'].items()))
    write(O/'CSDI_QUALIFICATION89_V1.json',receipt);log(dict(stage='csdi_qualified',**{k:v for k,v in receipt.items() if k!='utc'}))

if __name__=='__main__':main()
