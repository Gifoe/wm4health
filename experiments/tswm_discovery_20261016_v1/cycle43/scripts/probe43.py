import sys,pathlib,json,importlib.metadata as m
R=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(R.parent/'cycle29/scripts'));import runtime_profile
names=['torch','diffusers','transformers','einops','pytorch-lightning','lightning','hydra-core','omegaconf','timm','rotary-embedding-torch']
versions={}
for name in names:
 try:versions[name]=m.version(name)
 except m.PackageNotFoundError:versions[name]=None
print(json.dumps(versions))
