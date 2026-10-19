"""Fetch closest falsifiers of novelty, not just favorable related work."""
import importlib.util,pathlib
p=pathlib.Path(__file__).with_name('fetch_sources.py')
s=importlib.util.spec_from_file_location('fetch',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
for r in ['BenevolentAI/CoMP','Valentyn1997/CSM-APID','jessicalally/robust-cf-inference-beyond-mdps']:
    print(m.one(r),flush=True)
