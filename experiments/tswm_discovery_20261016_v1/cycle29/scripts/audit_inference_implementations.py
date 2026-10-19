"""Frozen validation-only algebraic and timing controls after both scientific queues."""
import runtime_profile
import copy, json, pathlib, sys, time
import numpy as np
import torch
from torch.nn import functional as F
from world_data import ROOT, OUT as DATA, sha, dump
from world_models_v2 import build
from train_world_comparison_v2 import SEEDS, KINDS, setup
from inference_implementations import prepare_fused_gru, fused_simulate, enable_unused_attention_weight_elision

OUT = ROOT / 'outputs/inference_controls_v1'


def measure(call):
    for _ in range(3): call()
    times = []
    for _ in range(9):
        torch.cuda.synchronize(); start = time.perf_counter(); call(); torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return dict(median_s=float(np.median(times)), p10_s=float(np.quantile(times, .1)), p90_s=float(np.quantile(times, .9)), n=9)


def difference(a, b):
    assert a.shape == b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
    torch.testing.assert_close(a, b, atol=2e-4, rtol=2e-4)
    return dict(max_absolute=float((a-b).abs().max()), relative_l2=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(a).clamp_min(1e-8)))


@torch.no_grad()
def cell_check(model, encoded, u):
    h, code = encoded
    gru, _ = prepare_fused_gru(model, encoded)
    if model.kind == 'compiled64':
        cell = torch.nn.GRUCell(1, model.width).to(h.device).eval()
        cell.weight_ih.copy_(model.cell.weight_ih); cell.weight_hh.copy_(model.cell.weight_hh + code[0])
        cell.bias_ih.copy_(model.cell.bias_ih); cell.bias_hh.copy_(model.cell.bias_hh)
        actual = cell(u[:, 0], h)
    else:
        actual = model.cell(torch.cat((u[:, 0], code), -1), h)
    expected, _ = gru(u[:, :1], h.unsqueeze(0))
    return difference(actual, expected[:, 0])


def csvwrite(path, rows):
    import csv
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader(); writer.writerows(rows)


@torch.no_grad()
def main():
    setup()
    assert json.loads((ROOT/'outputs/world_verification/model_comparison_v2.json').read_text())['status'] == 'passed'
    assert json.loads((ROOT/'outputs/real_transfer/verification.json').read_text())['status'] == 'passed'
    assert not OUT.exists(); OUT.mkdir()
    rows=[]; checks=[]; weights=[]
    for seed in SEEDS:
        for kind in KINDS:
            weight=ROOT/'outputs/world_training_v2'/f'{kind}_{seed}'/'best.pt'
            state=torch.load(weight,map_location='cpu',weights_only=True)
            model=build(kind).cuda().eval(); model.load_state_dict(state['model'],strict=True)
            weights.append(dict(kind=kind,seed=seed,sha256=sha(weight)))
            optimized=copy.deepcopy(model) if kind=='native_attention64' else None
            if optimized is not None: enable_unused_attention_weight_elision(optimized)
            for family in ['lti','wh']:
                path=DATA/f'val_{family}.npz'
                with np.load(path,allow_pickle=False) as data:
                    y=torch.tensor(data['y'][:2],device='cuda').unsqueeze(-1)
                    u=torch.tensor(data['u'][:2],device='cuda').unsqueeze(-1)
                for system in range(2):
                    cy=y[system:system+1,:128]; cu=u[system:system+1,:128]
                    for horizon in [32,128]:
                        for n in [1,16,64]:
                            future=torch.cat([torch.roll(u[system:system+1,128:128+horizon],shifts=i,dims=1) for i in range(n)])
                            encoded=model.encode(cy,cu)
                            eager_encoded=tuple(v.expand((n,)+v.shape[1:]).contiguous() for v in encoded)
                            expected=model.simulate(eager_encoded,future)
                            key=dict(kind=kind,seed=seed,family=family,system=system,horizon=horizon,n_plans=n)
                            if optimized is None:
                                step=cell_check(model,encoded,future[:1])
                                prepared=prepare_fused_gru(model,encoded)
                                replay=difference(expected,fused_simulate(model,prepared,future))
                                prep=measure(lambda:prepare_fused_gru(model,encoded))
                                sim=measure(lambda:fused_simulate(model,prepared,future))
                                enc=measure(lambda:model.encode(cy,cu))
                                variant='code_bias_fold_fused_gru' if kind!='compiled64' else 'merged_weight_fused_gru'
                            else:
                                opt_encoded=optimized.encode(cy,cu)
                                opt_encoded=tuple(v.expand((n,)+v.shape[1:]).contiguous() for v in opt_encoded)
                                replay=difference(expected,optimized.simulate(opt_encoded,future))
                                step=None; prep=dict(median_s=0.,p10_s=0.,p90_s=0.,n=0)
                                sim=measure(lambda:optimized.simulate(opt_encoded,future))
                                enc=measure(lambda:optimized.encode(cy,cu));variant='cached_memory_sdpa_no_weight_output'
                            eager_sim=measure(lambda:model.simulate(eager_encoded,future))
                            eager_enc=measure(lambda:model.encode(cy,cu))
                            checks.append(dict(**key,whole_sequence=replay,one_step=step))
                            for impl,e,s,p in [('source_eager',eager_enc,eager_sim,dict(median_s=0.,p10_s=0.,p90_s=0.,n=0)),(variant,enc,sim,prep)]:
                                row=dict(**key,implementation=impl,context_s=e['median_s'],prepare_s=p['median_s'],cached_s=s['median_s'],cached_p10_s=s['p10_s'],cached_p90_s=s['p90_s'],one_use_total_s=e['median_s']+p['median_s']+s['median_s'],amortized_64_uses_s=(e['median_s']+p['median_s'])/64+s['median_s'])
                                rows.append(row)
                    print(json.dumps(dict(phase='implementation_control',kind=kind,seed=seed,family=family,system=system,rows=len(rows))),flush=True)
            del model,optimized;torch.cuda.empty_cache()
    csvwrite(OUT/'latency.csv',rows);dump(OUT/'equivalence.json',checks)
    result=dict(status='passed',n_rows=len(rows),n_equivalence_checks=len(checks),weights=weights,latency_sha256=sha(OUT/'latency.csv'),equivalence_sha256=sha(OUT/'equivalence.json'),plan_sha256=sha(ROOT/'INFERENCE_IMPLEMENTATION_CONTROLS.md'),sources={name:sha(ROOT/'scripts'/name) for name in ['inference_implementations.py','audit_inference_implementations.py']},environment=dict(python=sys.version,torch=torch.__version__,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(),tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32),scope='validation-only finite-precision equivalence and implementation latency; no accuracy or universal lower-bound claim')
    dump(OUT/'results.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__': main()
