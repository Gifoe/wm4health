"""Generate paired sensors; keep inputs separate from diagnostic state labels."""
from common57 import *
import numpy as np

def main():
    pins=source_check();root=paths()['data_root'];root.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(5700);rho=.95;angles=np.array([-.2,.3]);records=[]
    matrices=np.array([[[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]] for a in angles])*rho
    for split,N in [('train',512),('validation',128),('heldout',256)]:
        u=rng.integers(0,2,size=(N,31),dtype=np.int64)
        z=np.empty((N,32,2));z[:,0]=rng.normal(size=(N,2));noise=rng.normal(size=(N,31,2))
        for t in range(31):z[:,t+1]=np.einsum('nij,nj->ni',matrices[u[:,t]],z[:,t])+np.sqrt(1-rho*rho)*noise[:,t]
        y=z+np.sqrt(.3)*rng.normal(size=z.shape)
        for kind,values in [('inputs',{'linear':y.astype('float32'),'asinh':np.arcsinh(y).astype('float32'),'actions':u}),('truth',{'z':z,'y':y,'rho':np.array(rho),'angles':angles,'observation_variance':np.array(.3)})]:
            p=root/f'{split}_{kind}_v1.npz'
            with p.open('xb') as f:np.savez(f,**values)
            records.append({'split':split,'kind':kind,'sequences':N,'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
    dump(OUT/'DATA57_V1.json',{'utc':utc(),'status':'PAIRED_SENSOR_CONTROLLED_SYSTEMS_GENERATED','seed':5700,'protocol_sha256':sha(ROOT/'PROTOCOL.md'),'script_sha256':sha(__file__),'source_files_sha256':pins,'files':records,'heldout_training_reads':0,'formal_updates':0})
    print(json.dumps({'status':'DATA_GENERATED','files':len(records),'bytes':sum(x['bytes'] for x in records)}),flush=True)

if __name__=='__main__':main()
