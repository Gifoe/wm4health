import sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common81_v1 import *
def main():
 manifest=read(PROJECT/'cycle80/outputs/ASSET_MANIFEST80_V1.json');asset=next(a for a in manifest['assets'] if a['relative'].endswith('strogatz_extended.json'));assert sha(asset['path'])==asset['sha256'];db=read(asset['path']);eligible=[]
 for e in db:
  if e['dim']!=2 or e['id'] in [26,42,28] or not e.get('solutions'):continue
  paths=[s for s in e['solutions'][0] if s.get('success',True)]
  if len(paths)>=2 and all(len(s['t'])==512 for s in paths[:2]):eligible.append(int(e['id']))
 eligible=sorted(eligible);dev=eligible[0::3][:6];reserved=eligible[1::3][:6];assert len(dev)==len(reserved)==6 and not set(dev)&set(reserved)
 dump(OUT/'COHORT81_V1.json',dict(utc=utc(),protocol_sha256=sha(ROOT/'PROTOCOL81_V1.md'),asset_sha256=asset['sha256'],eligible=eligible,development=dev,reserved_evaluation=reserved,selection_uses_signal_values=False,evaluation_opened=False,previously_scored_exclusions=[26,42,28],context_seed=2026810,context_noise_sigma=.03))
 print(json.dumps(read(OUT/'COHORT81_V1.json')),flush=True)
if __name__=='__main__':main()
