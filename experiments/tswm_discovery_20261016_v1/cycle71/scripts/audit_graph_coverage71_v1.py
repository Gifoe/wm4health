"""Measure graph overlap in the declared TRAIN/VAL rows; never inspect EVAL."""
import sys,pathlib,collections
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common71_v1 import *
import numpy as np

acq=read(OUT/'ACQUISITION71_V1.json');data=pathlib.Path(acq['data_root']);src=pathlib.Path(acq['source_root'])/'neurips2019'/'nri_structure.json'
spec=next(r for r in acq['sources'] if r['path']=='neurips2019/nri_structure.json');assert sha(src)==spec['sha256']
assert sha(data/'edges_springs5_train.npy')=='a99f886f4b057fb95372218d78e0902c51eab8f6a6c11d643311f53ab0805f12'
assert sha(data/'edges_springs5_test.npy')=='9b7d18a19fa6ad8c37a9d1b30b83712c77cb440466901409bd7d619c154fcfc3'
train=np.load(data/'edges_springs5_train.npy',mmap_mode='r',allow_pickle=False)[:2048].copy();val=np.load(data/'edges_springs5_test.npy',mmap_mode='r',allow_pickle=False)[:256].copy();assert train.shape==(2048,20) and val.shape==(256,20)
edges=read(src)['edges'];reverse=[edges.index([j,i]) for i,j in edges];undirected=[k for k,(i,j) in enumerate(edges) if i<j]
train_unique=np.unique(train,axis=0);val_unique=np.unique(val,axis=0);known={tuple(e) for e in train_unique};seen=np.array([tuple(e) in known for e in val]);seen_unique=sum(tuple(e) in known for e in val_unique);distance=np.min(np.sum(train_unique[None]!=val[:,None],axis=-1),axis=1)
path=storage('artifact')/'GRAPH_COVERAGE71_V1.npz';assert not path.exists();np.savez_compressed(path,val_graph_seen_in_train=seen,val_min_directed_hamming_distance=distance)
obj=dict(utc=utc(),status='TRAIN_VAL_GRAPH_OVERLAP_MEASURED',train_rows=2048,val_rows=256,eval_rows_accessed=0,train_unique_graphs=len(train_unique),val_unique_graphs=len(val_unique),val_rows_with_train_graph=int(seen.sum()),val_row_seen_graph_fraction=float(seen.mean()),val_unique_graphs_seen_in_train=int(seen_unique),val_unique_seen_graph_fraction=float(seen_unique/len(val_unique)),train_symmetric_graph_fraction=float(np.mean(np.all(train==train[:,reverse],axis=1))),val_symmetric_graph_fraction=float(np.mean(np.all(val==val[:,reverse],axis=1))),min_directed_hamming_histogram={str(int(k)):int(v) for k,v in zip(*np.unique(distance,return_counts=True))},train_number_label1_undirected_edges_histogram={str(int(k)):int(v) for k,v in zip(*np.unique(np.sum(train[:,undirected],axis=1),return_counts=True))},source_graph_config_sha256=sha(src),train_edges_sha256=sha(data/'edges_springs5_train.npy'),val_container_edges_sha256=sha(data/'edges_springs5_test.npy'),raw_artifact=dict(path=str(path),sha256=sha(path)),structurally_disjoint_protocol=False,new_method=False,accepted_final_directions=0)
dump(OUT/'GRAPH_COVERAGE71_V1.json',obj);print(json.dumps(obj),flush=True)
