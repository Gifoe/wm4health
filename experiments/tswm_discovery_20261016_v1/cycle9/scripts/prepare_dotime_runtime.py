import runtime_profile
import hashlib,importlib.util,json,pathlib,subprocess,sys
import requests
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';OUT.mkdir(exist_ok=True)
def main():
 target=ROOT/'vendor';target.mkdir(exist_ok=True);sys.path.insert(0,str(target))
 if not importlib.util.find_spec('pfns'):
  s=requests.Session();s.trust_env=False;r=s.get('https://pypi.org/pypi/pfns/json',timeout=(5,15));r.raise_for_status();meta=r.json()
  wheels=[x for x in meta['urls'] if x['filename'].endswith('py3-none-any.whl')];assert len(wheels)==1
  row=wheels[0];r=s.get(row['url'],timeout=(5,20));r.raise_for_status();raw=r.content;assert hashlib.sha256(raw).hexdigest()==row['digests']['sha256']
  wheel=ROOT/'data'/row['filename'];wheel.parent.mkdir(exist_ok=True);wheel.write_bytes(raw)
  (OUT/'pfns_dependency.json').write_text(json.dumps({'name':'pfns','version':meta['info']['version'],'wheel':row['filename'],'sha256':row['digests']['sha256'],'dependencies':meta['info'].get('requires_dist'),'no_dependency_mutations':True},indent=2),encoding='utf-8')
  subprocess.run([sys.executable,'-m','pip','install','--no-deps','--no-index','--target',str(target),str(wheel)],check=True)
 src=ROOT.parent/'upstream/dotime';sys.path.insert(0,str(src/'src'))
 from dotime.models.loader import load_dotpfn
 import torch
 path=ROOT/'data/do_over_time_pfn_best.pt'
 if path.exists():
  unsafe=torch.serialization.get_unsafe_globals_in_checkpoint(path);print('pickle_globals',unsafe,flush=True);assert not unsafe,'Do not execute arbitrary unreviewed checkpoint globals'
  ckpt=torch.load(path,weights_only=True,map_location='cpu');print('config',json.dumps(ckpt['config']),flush=True)
  model=load_dotpfn(path,device='cuda');print('strict native loader passed',sum(x.numel() for x in model.parameters()),flush=True)
  record={'source_pin':json.loads((src/'SOURCE_PIN.json').read_text()),'checkpoint_config':ckpt['config'],'unsafe_globals':unsafe,'model_parameters':sum(x.numel() for x in model.parameters()),'input_convention':model.input_convention,'model_n_max':model.n_max}
  (OUT/'dotime_runtime.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
if __name__=='__main__':main()
