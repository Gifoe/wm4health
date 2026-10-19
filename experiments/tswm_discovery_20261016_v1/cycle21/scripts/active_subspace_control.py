"""Old sensitivity-based reduction under all-output versus action-output target choices."""
import runtime_profile
import hashlib,json,pathlib,time
import numpy as np
import torch
from sklearn.preprocessing import PolynomialFeatures
import response_target_prototype as proto
p=proto.p
ROOT=pathlib.Path(__file__).resolve().parents[1];BASE=ROOT/'outputs/response_target_prototype';OUT=ROOT/'outputs/active_subspace_v1'
ACTIONS=np.asarray([[-1,-1],[-1,1],[1,-1],[1,1]],dtype='float32')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def bank(model,x):
    n=len(x);xx=torch.tensor(np.repeat(x,4,axis=0));uu=torch.tensor(np.tile(ACTIONS,(n,1)))
    with torch.no_grad():return model.predict(xx,uu).numpy().reshape(n,4,12)
def feature_fit(z,u,degree):
    library=PolynomialFeatures(degree=degree,include_bias=True);matrix=library.fit_transform(np.column_stack([z,u]));scale=np.maximum(np.sqrt(np.mean(matrix**2,axis=0)),1e-8)
    return library,matrix/scale,scale
def main():
    torch.set_num_threads(1);tick=time.perf_counter();assert not OUT.exists();OUT.mkdir(parents=True)
    fits=json.loads((BASE/'teacher_records.json').read_text());assert json.loads((BASE/'independent_verification.json').read_text())['pass'];rows=[]
    for family in proto.FAMILIES:
        data={n:np.load(BASE/'data'/(family+'_'+n+'.npz'),allow_pickle=False) for n in ['train','val','test']}
        ready={n:p.prep({k:s[k] for k in ['current','future','action','trajectory_id']},s['mean'],s['scale'],s['rotation']) for n,s in data.items()}
        for fit in [r for r in fits if r['family']==family]:
            teacher_file=BASE/'teachers'/fit['checkpoint'];assert sha(teacher_file)==fit['checkpoint_sha256'];state=torch.load(teacher_file,map_location='cpu',weights_only=True)
            teacher=p.Model('fullobs');teacher.load_state_dict(state['state']);teacher.eval()
            for parameter in teacher.parameters():parameter.requires_grad_(False)
            forecasts=bank(teacher,ready['train']['x'][:8192]);response=forecasts-forecasts.mean(1,keepdims=True);r=response.reshape(-1,12).astype('float64')
            eig,basis=np.linalg.eigh(r.T@r/len(r));order=np.argsort(eig)[::-1];eig=np.maximum(eig[order],0);basis=basis[:,order]
            rank=int(np.searchsorted(np.cumsum(eig)/eig.sum(),.95)+1)
            for kind,output_basis in [('all_outputs',np.eye(12)),('action_outputs',basis[:,:rank])]:
                start=time.perf_counter();U=torch.tensor(output_basis,dtype=torch.float32)
                def fun(x,u):return teacher.predict(x[None,:],u[None,:])[0]@U
                xs=torch.tensor(np.repeat(ready['train']['x'][:512],4,axis=0));us=torch.tensor(np.tile(ACTIONS,(512,1)))
                jac=torch.func.vmap(torch.func.jacrev(fun,argnums=0))(xs,us).detach().numpy().astype('float64')
                finite=[];h=.001
                for i in range(4):
                    columns=[]
                    for j in range(12):
                        lo=xs[i].clone();hi=xs[i].clone();lo[j]-=h;hi[j]+=h
                        columns.append(((fun(hi,us[i])-fun(lo,us[i]))/(2*h)).detach().numpy())
                    finite.append(np.column_stack(columns))
                gradient_error=float(np.max(np.abs(np.stack(finite)-jac[:4])));assert gradient_error<.002
                covariance=np.einsum('nki,nkj->ij',jac,jac)/len(jac);ev,W=np.linalg.eigh(covariance);order=np.argsort(ev)[::-1];ev=ev[order];W=W[:,order[:4]]
                xmean=ready['train']['x'].mean(0);z=(ready['train']['x']-xmean)@W;z_cov=np.cov(z,rowvar=False)
                eig_z,V=np.linalg.eigh(z_cov);whitening=V@np.diag(1/np.sqrt(eig_z))@V.T;transform=W@whitening
                encoded={n:((v['x']-xmean)@transform).astype('float64') for n,v in ready.items()}
                successor={n:((v['y']-xmean)@transform).astype('float64') for n,v in ready.items()};candidates=[];models=[]
                for degree in [1,2,3]:
                    library,design,scales=feature_fit(encoded['train'],ready['train']['u'],degree)
                    vd=library.transform(np.column_stack([encoded['val'],ready['val']['u']]))/scales
                    for ridge in [1e-6,1e-3,.1]:
                        penalty=np.eye(design.shape[1])*ridge;penalty[0,0]=0
                        coeff=np.linalg.solve(design.T@design/len(design)+penalty,design.T@successor['train']/len(design))
                        score=float(np.mean((vd@coeff-successor['val'])**2));candidates.append(dict(degree=degree,ridge=ridge,validation_latent_MSE=score));models.append((library,scales,coeff))
                best=int(np.argmin([v['validation_latent_MSE'] for v in candidates]));library,scales,coeff=models[best]
                # Privileged readout only; physical/gain labels never select the encoder or transition.
                design=np.column_stack([encoded['train'][:8192],np.ones(8192)]);targets=ready['train']['current_labels'][:8192]
                decoder=np.linalg.lstsq(design,targets,rcond=None)[0]
                def forecast(x,u):
                    z=(x-xmean)@transform;zd=library.transform(np.column_stack([z,u]))/scales;zp=zd@coeff
                    return np.column_stack([zp,np.ones(len(zp))])@decoder
                pred=forecast(ready['test']['x'],ready['test']['u']);current=np.column_stack([encoded['test'],np.ones(len(encoded['test']))])@decoder
                saved=data['test'];lo=saved['current'].copy();hi=lo.copy();lo[:,2]=.4;hi[:,2]=1.6;mean=saved['mean'];scale=saved['scale'];rotation=saved['rotation']
                lx=((lo-mean)/scale@rotation.T).astype('float32');hx=((hi-mean)/scale@rotation.T).astype('float32')
                modulation=forecast(hx,ready['test']['u'])[:,:2]-forecast(lx,ready['test']['u'])[:,:2]
                true_mod=(proto.update(hi[:,:2],hi[:,2],ready['test']['u'],family)-proto.update(lo[:,:2],lo[:,2],ready['test']['u'],family))/scale[:2]
                gain_accuracy=float(np.mean((current[:,2]*scale[2]+mean[2]>1)==(saved['current'][:,2]>1)))
                key=family+'_'+kind+'_'+str(fit['seed']);file=OUT/(key+'.npz')
                np.savez_compressed(file,input_transform=transform,input_mean=xmean,output_basis=output_basis,output_response_eigenvalues=eig,
                    input_gradient_eigenvalues=ev,feature_powers=library.powers_,feature_scales=scales,transition_coefficients=coeff,diagnostic_decoder=decoder,
                    prediction=pred,current_prediction=current,modulation=modulation,true_modulation=true_mod)
                row=dict(key=key,family=family,output_choice=kind,seed=fit['seed'],output_rank=output_basis.shape[1],input_dimension=4,
                    gain_accuracy=gain_accuracy,current_physical_decoding_MSE=float(np.mean((current[:,:2]-ready['test']['current_labels'][:,:2])**2)),
                    native_physical_MSE=float(np.mean((pred[:,:2]-ready['test']['future_labels'])**2)),
                    modulation_relative_RMSE=float(np.sqrt(np.mean((modulation-true_mod)**2)/np.mean(true_mod**2))),
                    finite_difference_gradient_max_error=gradient_error,selected=candidates[best],candidates=candidates,
                    model_file=file.name,model_sha256=sha(file),teacher_sha256=fit['checkpoint_sha256'],seconds=time.perf_counter()-start)
                rows.append(row);print(json.dumps(dict(completed=len(rows),key=key,gain_accuracy=gain_accuracy,response_error=row['modulation_relative_RMSE'])),flush=True)
    result=dict(rows=rows,seconds=time.perf_counter()-tick,script_sha256=sha(pathlib.Path(__file__)),plan_sha256=sha(ROOT/'ACTIVE_SUBSPACE_CONTROL.md'),
        scope='Independent-cohort classical sensitivity and polynomial reduction; action-output variant shares the proposed output-selection ingredient. Not new active-subspace theory or independent confirmation.',
        accepted_novel_directions=0,objective_achieved=False)
    (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(dict(completed=24,seconds=result['seconds'])),flush=True)
if __name__=='__main__':main()
