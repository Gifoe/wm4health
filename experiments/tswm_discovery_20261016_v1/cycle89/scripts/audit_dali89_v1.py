from common89_v1 import *
import ast,importlib.metadata as md,re

def main():
    manifest=read(O/'SOURCE_MANIFEST89_V1.json');source=Path(manifest['source_root']);functions=[];focused=[]
    for row in manifest['files']:
        p=source/row['path'];assert sha(p)==row['sha256'],row['path']
        if not p.name in ['agent.py','nets.py','configs.yaml'] and not 'counterfactual' in p.name:continue
        text=p.read_text(encoding='utf-8')
        if p.suffix=='.py':
            tree=ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in ['img_step','obs_step','imagine','observe','__init__','gen_dream']:
                    body=ast.get_source_segment(text,node)
                    if 'context' in body or node.name=='gen_dream':functions.append(dict(path=row['path'],name=node.name,line=node.lineno,args=ast.unparse(node.args),source=body))
        for i,line in enumerate(text.splitlines(),1):
            if re.search(r'grssm|ctx_encoder|context_encoder|dcontext|[Ss]hallow|[Dd]eep|ctx_pos|sigma_z',line):focused.append(dict(path=row['path'],line=i,text=line))
    weights=[r for r in manifest['tree_entries'] if r['type']=='blob' and re.search(r'\.(ckpt|pt|pth|safetensors|pkl|pickle|npz)$',r['path'],re.I)]
    versions={}
    for name in ['jax','jaxlib','mujoco','dm-control','gymnasium','carl-bench','PyYAML','torch']:
        try:versions[name]=md.version(name)
        except md.PackageNotFoundError:versions[name]=None
    hf=read(source/'__metadata_3.json');hf_ids=[r.get('id','') for r in hf] if isinstance(hf,list) else []
    official_ids=[x for x in hf_ids if re.search(r'frankroeder|dreamer|world.?model|contextual',x,re.I)]
    receipt=dict(utc=utc(),status='PINNED_SOURCE_INTERFACES_AUDITED_NOT_MODEL_EXECUTION',commit=manifest['commit'],source_files_verified=len(manifest['files']),available_versions=versions,functions=functions,focused_lines=focused,repository_weight_files=weights,official_hf_search_candidates=official_ids,hf_search_truncated_or_limited=len(hf_ids)>=100,checkpoint_absence_proven=False,model_executed=False,scientific_training_fits=0)
    write(O/'DALI_SOURCE_AUDIT89_V1.json',receipt)
    print(json.dumps(dict(versions=versions,weight_files=weights,hf_candidates=official_ids,functions=[dict(path=f['path'],name=f['name'],line=f['line'],args=f['args']) for f in functions]),indent=2),flush=True)
    for f in functions:
        if f['name'] in ['img_step','obs_step'] or (f['name']=='__init__' and f['path'].endswith('agent.py')):print(json.dumps(f,indent=2),flush=True)

if __name__=='__main__':main()
