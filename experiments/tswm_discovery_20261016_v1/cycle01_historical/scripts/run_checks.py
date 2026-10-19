import json,pathlib,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
start=time.perf_counter()
p=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-v'],cwd=ROOT,capture_output=True,text=True)
(ROOT/'outputs/verification.log').write_text(p.stdout+p.stderr,encoding='utf-8')
(ROOT/'outputs/verification.json').write_text(json.dumps({'exit_code':p.returncode,'seconds':time.perf_counter()-start,'command':'python -m unittest discover -s tests -v'},indent=2),encoding='utf-8')
print(p.stdout+p.stderr);sys.exit(p.returncode)
