from common65 import *
p=paths()['checkpoint_root'];print(json.dumps(dict(files=[dict(name=x.name,bytes=x.stat().st_size) for x in p.glob('nonlinear_s2022*')],original_exit=read(OUT/'train_exit65_v1.json'),closed_models=len(read(OUT/'fit_registry65_v1.json')['fits']))))
