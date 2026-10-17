"""Describe raw rows and source transformations without fitting or selecting models."""
from common59 import *
import ast,importlib.util,numpy as np
def stats(a):
    return {'min':float(a.min()),'max':float(a.max()),'mean':float(a.mean()),'std':float(a.std()),'percentiles':np.percentile(a,[0,1,25,50,75,99,100]).tolist()}
def main():
    provenance=read(OUT/'DOWNLOAD59_V2.json')
    for f in provenance['files']:assert sha(f['server_path'])==f['sha256']
    arrays={k:np.loadtxt(data_root()/f'N87_5_{k}.csv.gz',delimiter=',') for k in ['B','H','T']}
    b,h,t=arrays['B'],arrays['H'],arrays['T'];t=np.atleast_1d(t)
    assert b.ndim==h.ndim==2 and b.shape==h.shape and len(t)==len(b)
    assert all(np.isfinite(a).all() for a in arrays.values())
    tree=ast.parse((source_root()/'run_1_dataset.py').read_text())
    dt=[n.value.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='dt' for x in n.targets)]
    assert dt==[62.5e-9]
    parser=(source_root()/'odesolver/parsing.py').read_text()
    preprocessing={'whole_row_B_centering':'B_mat -= np.mean(B_mat, axis=1, keepdims=True)' in parser,'whole_row_H_centering':'H_mat -= np.mean(H_mat, axis=1, keepdims=True)' in parser,'H_dependent_phase_alignment':'np.argmin(np.abs(H_mat), axis=1)' in parser,'circular_roll':'np.roll(row, -pos)' in parser,'periodic_interpolation':'period=t_period' in (source_root()/'odesolver/compute/dataset.py').read_text()}
    assert all(preprocessing.values())
    rowids=[]
    for i in range(len(b)):
        bh=hashlib.sha256(b[i].astype('<f8').tobytes()+h[i].astype('<f8').tobytes()+np.asarray(t[i],dtype='<f8').tobytes()).hexdigest()
        rowids.append({'row':i,'identity':f"{provenance['source_commit']}:N87_5:{i}",'sha256_B_H_T':bh,'temperature':float(t[i]),'points':b.shape[1]})
    # Chronology and time metadata derive from the pinned source, not tensor magnitudes.
    obj={'utc':utc(),'status':'RAW_MEASUREMENT_AND_SOURCE_PREPROCESSING_AUDITED_NO_MODEL_FIT','source_commit':provenance['source_commit'],'download_receipt_sha256':sha(OUT/'DOWNLOAD59_V2.json'),'shape_B_H':list(b.shape),'shape_T':list(t.shape),'sampling_interval_seconds':dt[0],'sampling_rate_hz':1/dt[0],'clock_evidence':'pinned run_1_dataset.py: dt assignment','time_extent_seconds':(b.shape[1]-1)*dt[0],'temperature_counts':{str(v):int(np.sum(t==v)) for v in np.unique(t)},'stats':{k:stats(a) for k,a in arrays.items()},'duplicate_B_H_T_rows':len(rowids)-len({r['sha256_B_H_T'] for r in rowids}),'raw_row_identities':rowids,'source_preprocessing':preprocessing,'source_preprocessing_verdict':'Tutorial periodic pipeline is not reused for boundary-conditioned causal deployment: full-row H centering and H-based circular phase alignment use response values outside a prediction prefix. This is a task-protocol difference, not proof of an author error.','unverified_metadata':['exact raw release equivalence to PI-HNO 14-material data','device/core ID per row','reset/prehistory','measurement repeatability','exact frequency identity from filename alone'],'single_material_only':True,'new_model_fits':0,'new_method_accepted':False,'local_result_files_written':0,'script_sha256':sha(__file__),'runtime_probe':{'numpy':np.__version__,'packages':{k:bool(importlib.util.find_spec(k)) for k in ['scipy','h5py','torch','jax','equinox','diffrax','optimistix','pandas']}},'overall_goal':'ACTIVE_UNACHIEVED','accepted_final_directions':0}
    dump(OUT/'DATA_AUDIT59_V2.json',obj)
    print(json.dumps({k:obj[k] for k in ['status','shape_B_H','sampling_interval_seconds','temperature_counts','duplicate_B_H_T_rows','source_preprocessing','runtime_probe']}),flush=True)
if __name__=='__main__':main()
