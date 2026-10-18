"""Add only the pinned project-local CPU prefix to the original audit runtime."""
from common72_v1 import *
DEPENDENCIES=read(OUT/'DEPENDENCIES72_V1.json')
assert DEPENDENCIES['status']=='SCOPED_CPU_DEPENDENCIES_INSTALLED_WITH_SHARED_NUMPY_TORCH_UNCHANGED'
prefix=pathlib.Path(DEPENDENCIES['target']).resolve()
assert prefix.drive.upper()=='G:' and prefix==((storage('cache')/'cpu_dependencies_v1').resolve())
sys.path.insert(0,str(prefix))
