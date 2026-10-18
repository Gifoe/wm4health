from common88_v1 import *
import ast,importlib,yaml,torch

M=read(O/'SOURCE_MANIFEST88_V1.json')
SOURCE=Path(M['source_root'])
sys.path.insert(0,str(SOURCE))

class AttrDict(dict):
    def __getattr__(self,k):
        try:return self[k]
        except KeyError:raise AttributeError(k) from None

def attrs(x):
    if isinstance(x,dict):return AttrDict({k:attrs(v) for k,v in x.items()})
    if isinstance(x,list):return [attrs(v) for v in x]
    return x

def model_config(arm):
    kind='acrkn' if arm=='acrkn' else 'mts3'
    c=yaml.safe_load((SOURCE/f'experiments/mobileRobot/conf/model/default_{kind}.yaml').read_text(encoding='utf-8'))
    c['wandb']['log']=False
    if kind=='mts3':c['mts3']['time_scale_multiplier']=32
    return attrs(c)

def make_model(arm,seed=2026884):
    assert arm in ['stock','lagged','acrkn']
    torch.manual_seed(seed)
    if arm=='acrkn':
        from agent.worldModels.acRKN import acRKN
        cls=acRKN
    else:
        module=importlib.import_module('agent.worldModels.MTS3')
        cls=module.MTS3
        if arm=='lagged':
            p=SOURCE/'agent/worldModels/MTS3.py'
            original=p.read_text(encoding='utf-8')
            tree=ast.parse(original);changed=0
            for node in ast.walk(tree):
                if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='current_act_seqs':
                    if ast.unparse(node.value)=='action_seqs[:, k + self.H:k + 2 * self.H, :]':
                        node.value=ast.parse('action_seqs[:, k:k+self.H, :] if k+self.H < obs_seqs.shape[1] else action_seqs[:, k+self.H:k+2*self.H, :]',mode='eval').body
                        changed+=1
            assert changed==1,changed
            ast.fix_missing_locations(tree);namespace=dict(vars(module))
            exec(compile(tree,'<one-index-lagged-MTS3>','exec'),namespace)
            cls=namespace['MTS3']
    model=cls(input_shape=(16,),action_dim=2,config=model_config(arm),use_cuda_if_available=True).cuda()
    return model

def qualify_sources():
    for row in M['files']:assert sha(SOURCE/row['path'])==row['sha256'],row['path']

def tensor_state_sha(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
