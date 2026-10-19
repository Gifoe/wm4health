import runtime_profile
import json,pathlib
import torch
import real_models as m
def main():
    m.setup();data=m.load_split('train');x,u,y=[data[k][:8] for k in ['x','u','y']];results=[]
    for method in m.METHODS:
        outcomes=[]
        for repeat in range(2):
            torch.manual_seed(4242);model=m.make(method);stage='latent' if method.startswith('worldts') and not method.endswith('endtoend') else 'supervised'
            m.mode(model,stage,True);optim=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001);criterion=m.latent_loss() if stage=='latent' else None
            for i in range(3):
                score,_=m.loss(model,stage,x,u,y,criterion);optim.zero_grad(set_to_none=True);score.backward();optim.step()
            if stage=='latent':
                before={k:v.clone() for k,v in model.state_dict().items() if not k.startswith('decoder.')};m.mode(model,'decoder',True)
                optim=torch.optim.Adam(model.decoder.parameters(),lr=.001);score,_=m.loss(model,'decoder',x,u,y,None);optim.zero_grad(set_to_none=True);score.backward();optim.step()
                assert all(torch.equal(value,model.state_dict()[key]) for key,value in before.items())
            model.eval()
            with torch.no_grad():pred=model.forecast(x,x_future=u)
            assert pred.shape==(8,16,11) and torch.isfinite(pred).all();outcomes.append(pred.cpu())
        error=float(torch.max(torch.abs(outcomes[0]-outcomes[1])));assert error==0
        results.append(dict(method=method,same_seed_three_updates_replay_error=error,output_shape=list(pred.shape)))
    file=m.ROOT/'outputs/preflight.json';assert not file.exists();file.write_text(json.dumps(dict(pass_=True,rows=results),indent=2));print(json.dumps(results),flush=True)
if __name__=='__main__':main()
