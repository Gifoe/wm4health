"""Independent NumPy metric arithmetic and same-seed native prefix replay."""
import runtime_profile
import hashlib,json,pathlib,sys,time
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from safetensors.torch import load_file
ROOT=pathlib.Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    started=time.perf_counter();torch.set_num_threads(1)
    folder=ROOT/'outputs/native_demo';result=json.loads((folder/'results.json').read_text())
    receipt=folder/'independent_verification.json';assert not receipt.exists()
    errors={};max_relative=0.;data={}
    for name in ['lorenz63','chua','sprottM']:
        saved=np.load(folder/(name+'.npz'),allow_pickle=False);data[name]=saved
        assert sha(folder/(name+'.npz'))==result['arrays_sha256'][name+'.npz']
        cl=int(saved['context_length']);truth=saved['data'][cl:cl+10000].astype('float32')
        lo=truth.min(0);hi=truth.max(0)
        def histogram(x):
            coord=(30*(x-lo)/(hi-lo)).astype('int64');mask=np.all((coord>0)&(coord<30),1)
            flat=np.ravel_multi_index(coord[mask].T,(30,30,30))
            hist=np.bincount(flat,minlength=27000).astype('float64');return hist,mask.mean()
        th,tr=histogram(truth);tp=(th+1e-5)/(th.sum()+1e-5*27000)
        for row in [r for r in result['rows'] if r['system']==name]:
            x=saved['predictions'][list(saved['seeds']).index(row['seed'])] if 'seed' in row else saved[row['control']]
            hist,retention=histogram(x);pdf=(hist+1e-5)/(hist.sum()+1e-5*27000)
            mse_den=np.abs(np.diff(truth.astype('float64'),axis=0)).mean(0)
            x_den=np.abs(np.diff(x.astype('float64'),axis=0)).mean(0)
            numerator=np.abs(x[:10].astype('float64')-truth[:10]).mean(0)
            expected=row['metrics'];estimates={
                'D_stsp_native':float(np.sum(tp*np.log(tp/pdf))) if hist.sum()>0 else None,
                'MASE_native_notebook_argument_order':float(np.mean(numerator/(x_den+1e-8))),
                'MASE_true_denominator_diagnostic':float(np.mean(numerator/(mse_den+1e-8))),
                'native_histogram_prediction_retention':float(retention),'native_histogram_truth_retention':float(tr),
                'truth_bbox_escape_fraction':float(np.any((x<lo)|(x>hi),axis=1).mean())}
            if expected['D_H_native'] is not None:
                spectra=[]
                for series in [truth,x]:
                    a=series.astype('float64');a-=a.mean(0)
                    power=np.abs(np.fft.rfft(a,axis=0))**2
                    smooth=gaussian_filter1d(power,20,axis=0);spectra.append(smooth/smooth.sum(0))
                estimates['D_H_native']=float(np.mean(np.sqrt(np.maximum(0,1-np.sqrt(spectra[0]*spectra[1]).sum(0)))))
            relative={}
            for key,value in estimates.items():
                if value is None:assert expected[key] is None;continue
                e=abs(value-expected[key]);relative[key]=e/(1+abs(value))
                assert relative[key]<2e-5,(name,row.get('seed',row.get('control')),key,value,expected[key],e)
            errors[name+'_'+str(row.get('seed',row.get('control')))]=max(relative.values());max_relative=max(max_relative,max(relative.values()))
    source=ROOT/'upstream/DynaMix-python';sys.path.insert(0,str(source/'src'))
    from dynamix.model.dynamix import DynaMix
    from dynamix.model.forecaster import DynaMixForecaster
    cfg=json.loads((ROOT/'outputs/external/DynaMix/config_3d-alrnn-v1.0.json').read_text())['architecture']
    model=DynaMix(**cfg).eval();model.load_state_dict(load_file(str(ROOT/'outputs/external/DynaMix/dynamix-3d-alrnn-v1.0.safetensors')),strict=True)
    saved=data['lorenz63'];cl=int(saved['context_length']);context=torch.tensor(saved['data'][:cl],dtype=torch.float32)
    torch.manual_seed(2021);prefix=DynaMixForecaster(model).forecast(context,64,standardize=True).numpy()
    replay_error=float(np.max(np.abs(prefix-saved['predictions'][0,:64])));assert replay_error==0
    for name in ['manifest.json','demo_manifest.json']:
        manifest=json.loads((ROOT/'outputs/source/DynaMix'/name).read_text())
        for row in manifest['files']:assert sha(source/row['path'])==row['sha256']
    info={'pass':True,'NumPy_metric_relative_errors':errors,'maximum_relative_metric_error':max_relative,
        'native_lorenz_same_seed64_prefix_error':replay_error,'source_and_demo_files_unchanged':True,
        'verified_arrays_sha256':result['arrays_sha256'],'script_sha256':sha(pathlib.Path(__file__)),
        'elapsed_seconds':time.perf_counter()-started,'objective_achieved':False}
    receipt.write_text(json.dumps(info,indent=2),encoding='utf-8');print(json.dumps(info),flush=True)
if __name__=='__main__':main()
