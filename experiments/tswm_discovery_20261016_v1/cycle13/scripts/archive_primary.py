"""Versioned primary-source archive for acquisition-policy opportunity mapping."""
import hashlib,json,pathlib
import requests
ROOT=pathlib.Path(__file__).resolve().parents[1]
PAPERS=[('timeflies','2606.13571v1'),('missingness_shift','2406.16484v1'),('mirrams','2507.08280v2'),
        ('under_cali','2605.28603v1'),('latent_ode','1907.03907v1'),('ossm','2609.33566v1')]

def main():
    out=ROOT/'outputs/literature';out.mkdir(parents=True,exist_ok=True);rows=[];errors=[]
    s=requests.Session();s.trust_env=False
    for key,rev in PAPERS:
        url='https://arxiv.org/pdf/'+rev;p=out/(key+'.pdf')
        try:
            if not p.exists():
                r=s.get(url,timeout=(5,30));r.raise_for_status();assert r.content.startswith(b'%PDF') and len(r.content)<15_000_000;p.write_bytes(r.content)
            raw=p.read_bytes();assert raw.startswith(b'%PDF')
            rows.append({'key':key,'version':rev,'url':url,'path':p.relative_to(ROOT).as_posix(),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()});print(key,len(raw),flush=True)
        except (OSError,requests.RequestException,AssertionError) as e:
            errors.append({'key':key,'url':url,'error':repr(e)});print('archive_error',key,type(e).__name__,flush=True)
    (out/'registry.json').write_text(json.dumps({'papers':rows,'errors':errors},indent=2),encoding='utf-8')

if __name__=='__main__':main()
