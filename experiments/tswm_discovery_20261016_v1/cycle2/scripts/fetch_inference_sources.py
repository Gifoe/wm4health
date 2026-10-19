"""Save primary-source snapshots with provenance; no paper claims from snippets."""
import datetime,hashlib,json,pathlib,requests
from pypdf import PdfReader
ROOT=pathlib.Path(__file__).resolve().parents[1]
SOURCES={
 'rp_gssm_icml2026':'https://raw.githubusercontent.com/mlresearch/v306/main/assets/hromadka26a/hromadka26a.pdf',
 'rpm_aistats2023':'https://proceedings.mlr.press/v206/walker23a/walker23a.pdf',
 'ramp_probml2026':'https://raw.githubusercontent.com/mlresearch/v327/main/assets/fox26a/fox26a.pdf',
 'rp_lds_2025':'https://discovery.ucl.ac.uk/id/eprint/10198967/1/RP_LDS_camera_ready.pdf',
}
def main():
 out=ROOT/'outputs/literature';out.mkdir(exist_ok=True,parents=True);records=[]
 session=requests.Session();session.trust_env=False
 for name,url in SOURCES.items():
  try:
   r=session.get(url,timeout=45);r.raise_for_status();assert r.content.startswith(b'%PDF')
   path=out/(name+'.pdf');path.write_bytes(r.content)
   pdf=PdfReader(path);txt='\n'.join(f'\n[PDF PAGE {i+1}]\n'+p.extract_text() for i,p in enumerate(pdf.pages))
   (out/(name+'.txt')).write_text(txt,encoding='utf-8')
   records.append({'name':name,'url':url,'final_url':r.url,'sha256':hashlib.sha256(r.content).hexdigest(),'pages':len(pdf.pages),'fetched_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
  except Exception as e:records.append({'name':name,'url':url,'error':repr(e)})
  print(json.dumps(records[-1]),flush=True)
 (out/'source_manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
if __name__=='__main__':main()
