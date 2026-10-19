import runtime_profile
import hashlib,json,pathlib
import innovation_pilot as p
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);assert torch.cuda.is_available()
    checks=p.numerical_checks();r=torch.Generator(device='cuda').manual_seed(22001)
    x=torch.randn(128,12,generator=r,device='cuda');y=torch.randn(128,12,generator=r,device='cuda');u=torch.sign(torch.randn(128,2,generator=r,device='cuda'))
    errors={}
    for method in p.METHODS:
        states=[]
        for repeat in range(2):
            torch.manual_seed(2021);model=p.Model(method).to('cuda');optimizer=torch.optim.Adam(model.parameters(),lr=.001)
            for _ in range(3):
                loss,_=p.losses(model,x,y,u);optimizer.zero_grad();loss.backward();optimizer.step();assert torch.isfinite(loss)
            states.append(torch.cat([v.detach().flatten() for v in model.parameters()]))
        error=float((states[0]-states[1]).abs().max());assert error==0;errors[method]=error
    receipt=ROOT/'outputs/preflight.json';receipt.parent.mkdir(parents=True,exist_ok=True);assert not receipt.exists()
    result=dict(pass_=True,numerical_checks=checks,three_update_same_seed_replay_errors=errors,
        torch=torch.__version__,gpu=torch.cuda.get_device_name(),script_sha256=hashlib.sha256((ROOT/'scripts/innovation_pilot.py').read_bytes()).hexdigest())
    result['pass']=result.pop('pass_');receipt.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
