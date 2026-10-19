import json,pathlib
from scipy.io import loadmat
p=loadmat(pathlib.Path(__file__).resolve().parents[1]/'outputs/data/CED/DATAUNIF.MAT')
print(json.dumps({k:list(v.shape) for k,v in p.items() if not k.startswith('__')}))
