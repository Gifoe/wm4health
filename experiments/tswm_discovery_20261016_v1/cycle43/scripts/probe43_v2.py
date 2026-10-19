import sys,pathlib,json,importlib.metadata as m
R=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(R.parent/'cycle29/scripts'));import runtime_profile
names=['torch','torchvision','huggingface-hub','packaging','filelock','safetensors','importlib_metadata','tensorboardX','IPython','numpy','Pillow','requests','regex','PyYAML','fsspec','tqdm','protobuf','transformers','diffusers']
versions={}
for name in names:
 try:versions[name]=m.version(name)
 except m.PackageNotFoundError:versions[name]=None
print(json.dumps(versions))
