"""Read current task runtime/cache without loading any model."""
import pathlib,sys,json,importlib.metadata as m,subprocess
packages={}
for name in ['torch','transformers','huggingface-hub','safetensors','tokenizers','numpy','pandas','pyarrow','scipy']:
    try:packages[name]=m.version(name)
    except m.PackageNotFoundError:packages[name]=None
roots=[pathlib.Path('D:/tswm_discovery_20261016_v1'),pathlib.Path('C:/Users/fyl412/.cache/huggingface/hub')]
found=[]
for root in roots:
    if root.exists():
        # Only named small top-level cache/task directories; no whole-disk walk.
        found += [str(p) for p in root.iterdir() if 'sundial' in p.name.lower()]
print(json.dumps({'python':sys.executable,'versions':packages,'sundial_top_level_paths':found,
   'gpu':subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],
          capture_output=True,text=True).stdout.strip()}))
