"""FIT-independent numeric fixtures for the separate post-run scorer."""
import pathlib,ast,numpy as np,json,hashlib
from common37_v2 import scores
R=pathlib.Path(__file__).resolve().parents[1];source=R/'scripts/verify_learning37.py'
tree=ast.parse(source.read_text(encoding='utf-8'));fns=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in ['crps','recompute']]
assert len(fns)==2;ns={'np':np};exec(compile(ast.Module(body=fns,type_ignores=[]),str(source),'exec'),ns)
rng=np.random.default_rng(9201);fixtures=[];maximum=0
for n in [2,8,128]:
    for mode in ['random','all_equal','duplicates']:
        s=rng.normal(size=(3,n,24));y=rng.normal(size=(3,24))
        if mode=='all_equal':s[:]=.1
        if mode=='duplicates':s[:,n//2:]=s[:,:n-n//2]
        a=scores(s,y);b=ns['recompute'](s,y)
        err=max(float(np.max(np.abs(a[k]-b[k]))) for k in a);maximum=max(maximum,err);assert err<2e-8,(n,mode,err)
        fixtures.append({'paths':n,'mode':mode,'max_abs':err})
out={'passed':True,'fixtures':fixtures,'max_abs':maximum,'no_benchmark_val_or_test_scores':True,
    'verifier_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
path=R/'outputs/verifier_numeric_preflight_v1.json';assert not path.exists();path.write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({'passed':True,'fixtures':len(fixtures),'max_abs':maximum}))
