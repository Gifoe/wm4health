"""Private dependency metadata and SHA-checked source-data window construction."""
import pathlib,sys,json,hashlib,zipfile,datetime
R=pathlib.Path(__file__).resolve().parents[1];ROOT=R.parent
sys.path.insert(0,str(ROOT/'cycle29/scripts'));import runtime_profile
import numpy as np
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def main():
 assert not (R/'data').exists();vendor=R/'vendor';vendor.mkdir(exist_ok=True);w=R/'assets/jaxtyping-0.2.36-py3-none-any.whl';assert sha(w)=='b19bcbd4009df8734602203402483a4066ad2eb3382904432e370588e9c9707d'
 with zipfile.ZipFile(w) as z:
  for member in z.infolist():
   if member.is_dir():continue
   parts=pathlib.PurePosixPath(member.filename).parts;assert '..' not in parts and not member.filename.startswith('/');p=vendor.joinpath(*parts);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(member))
 dist=vendor/'torchdde-0.2.0.dist-info';dist.mkdir(exist_ok=True);(dist/'METADATA').write_text('Metadata-Version: 2.1\nName: torchdde\nVersion: 0.2.0\nSummary: Private metadata for unchanged article source vendoring; not a published wheel\n',encoding='utf-8');dump(vendor/'VENDORING.json',{'jaxtyping_wheel_sha256':sha(w),'torchdde_source':'../source_archive/torchdde','metadata_only_for_importlib_version':True,'global_installation':False})
 for name in ['learnable_delays','torchdde']:
  src=R/'source_archive'/name
  for ent in json.loads((src/'SOURCE_TREE.json').read_text())['files']:assert sha(src/ent['path'])==ent['sha256']
 original=json.loads((ROOT/'cycle40/outputs/intake.json').read_text());out=R/'data';out.mkdir();records={};description={}
 for case in ['ett','jena']:
  folder=out/case;folder.mkdir();info=original['data'][case];source=ROOT/'cycle40/assets'/(case+'.npz');assert sha(source)==info['array_sha256'];raw=np.load(source);signal=raw['normalized'];timestamps=raw['timestamps'];assert np.all(np.diff(timestamps)==3600*10**9);phase=((timestamps//(3600*10**9))%24)/24.;assert np.allclose(signal,(raw['y']-info['train_mean'])/info['train_std'],equal_nan=True);description[case]={'field':info['field'],'unit':info['unit'],'source_array_sha256':sha(source),'source_raw_files':original['retrieval']['sources'],'train_mean':info['train_mean'],'train_std':info['train_std'],'splits':{}}
  for split,step in [('train',3),('val',48),('eval',48)]:
   spec=info['splits']['test' if split=='eval' else split];origins=np.arange(spec['start']+96,spec['stop_exclusive']-48+1,step);origins=np.array([o for o in origins if np.isfinite(signal[o-96:o+48]).all()],dtype='int64');assert len(origins)>5;history=np.stack([signal[o-96:o] for o in origins]).astype('float32')[:,:,None];future=np.stack([signal[o:o+48] for o in origins]).astype('float32')[:,:,None];np.savez_compressed(folder/(split+'.npz'),history=history,future=future,phase=phase[origins-1].astype('float32'),origins=origins,history_timestamps=np.stack([timestamps[o-96:o] for o in origins]),future_timestamps=np.stack([timestamps[o:o+48] for o in origins]));description[case]['splits'][split]={'origins':len(origins),'stride':step,'start':spec['start'],'stop_exclusive':spec['stop_exclusive']}
 for p in out.rglob('*.npz'):records[p.relative_to(out).as_posix()]={'sha256':sha(p),'bytes':p.stat().st_size}
 dump(out/'MANIFEST.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'FROZEN_NEW_MEMORY_PROTOCOL','files':records,'description':description,'HISTORY':96,'FUTURE':48,'protocol_sha256':sha(R/'PROTOCOL.md'),'source_sha256':sha(pathlib.Path(__file__)),'C40_test_metrics_reused':False});print(json.dumps({'status':'FROZEN_NEW_MEMORY_PROTOCOL','data':description}),flush=True)
if __name__=='__main__':main()
