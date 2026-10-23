"""Load pinned renderer Python unchanged, with a qualified forward-only CUDA binding.

Kaolin's original OBJ import function is executed unchanged. For the exercised
default no-material/no-normal branch only, SurfaceMesh is an identity container.
This is a restricted loader, not an installation or full reimplementation of Kaolin.
"""
import pathlib,types,sys,ast,hashlib,importlib.util,torch,warnings
from nvrtc_native_forward_v2 import NativeForward

def file_module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def load(root):
 root=pathlib.Path(root)
 acq=__import__('json').loads((root/'renderer_acquisition_v1/RESULTS_V1.json').read_text())
 src=pathlib.Path(acq['source_root'])
 supplement=__import__('json').loads((root/'renderer_source_supplement_v1/RESULTS_V1.json').read_text())
 for row in acq['files']:assert hashlib.sha256((src/row['path']).read_bytes()).hexdigest()==row['sha256']
 for row in supplement['files']:assert hashlib.sha256(pathlib.Path(row['cache_path']).read_bytes()).hexdigest()==row['sha256']
 backend=NativeForward(src)
 def package(name,path=None):
  assert name not in sys.modules,('Refuse replacing an installed module',name)
  m=types.ModuleType(name);m.__path__=[str(path)] if path else [];sys.modules[name]=m;return m
 grad=package('gradsim',src/'gradsim');utils=package('gradsim.utils',src/'gradsim/utils');grad.utils=utils
 utils.meshutils=file_module('gradsim.utils.meshutils',src/'gradsim/utils/meshutils.py')
 render=package('gradsim.renderutils',src/'gradsim/renderutils');grad.renderutils=render
 cuda=package('gradsim.renderutils.cuda');cuda.soft_rasterize_cuda=backend
 file_module('gradsim.renderutils.lighting',src/'gradsim/renderutils/lighting.py')
 raster=file_module('gradsim.renderutils.soft_rasterize',src/'gradsim/renderutils/soft_rasterize.py')
 renderer=file_module('gradsim.renderutils.softrenderer',src/'gradsim/renderutils/softrenderer.py');render.SoftRenderer=renderer.SoftRenderer
 objfile=pathlib.Path(next(x['cache_path'] for x in supplement['files'] if x['path']=='kaolin/io/obj.py'))
 source=objfile.read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='import_mesh')
 def error(*args,**kwargs):raise AssertionError('Unexercised OBJ error/material path')
 class IdentityMesh:
  def __init__(self,**kw):self.__dict__.update(kw)
 ns=dict(torch=torch,SurfaceMesh=IdentityMesh,default_error_handler=error,warnings=warnings)
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(objfile),'exec'),ns)
 native_import=ns['import_mesh']
 def restricted_import(path,**kwargs):
  assert not kwargs,('Only native default OBJ reader arguments qualified',kwargs)
  lines=pathlib.Path(path).read_text().splitlines();faces=[l.split()[1:] for l in lines if l.startswith('f ')]
  assert faces and all(len(f)==3 and all(int(x.split('/')[0])>0 for x in f) for f in faces)
  return native_import(path)
 kaolin=package('kaolin');kaolin.io=package('kaolin.io');kaolin.io.obj=types.SimpleNamespace(import_mesh=restricted_import)
 meta=pathlib.Path(r'G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\caches\meta_ilad_native_20261023_v1\source')
 decoder=file_module('_pinned_meta_pendulum_renderer',meta/'pendulum/gen_data/renderer.py')
 return types.SimpleNamespace(backend=backend,soft_rasterize=raster.soft_rasterize,SoftRenderer=renderer.SoftRenderer,PhysicsDecoder=decoder.PhysicsDecoder,mesh_import=restricted_import,meta_source=meta,gradsim_source=src,obj_import_ast_sha256=hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest())
