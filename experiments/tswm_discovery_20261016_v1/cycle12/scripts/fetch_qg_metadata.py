"""Read official checkpoint metadata, without downloading signal/model archives."""
import datetime,hashlib,json,pathlib,socket
import requests
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';OUT.mkdir(parents=True,exist_ok=True)
def main():
 s=requests.Session();s.trust_env=False;url='https://zenodo.org/api/records/19335706'
 try:
  r=s.get(url,timeout=(5,20));r.raise_for_status();record=r.json();(OUT/'qg_checkpoint_record.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
  rows=[{'key':r['key'],'size':r['size'],'checksum':r['checksum']} for r in record['files']];print(json.dumps({'official_record_files':rows}),flush=True)
  for item in record['files']:
   if item['key']=='models_archive_description.md':
    r=s.get(item['links']['self'],timeout=(5,20));r.raise_for_status();raw=r.content;assert len(raw)<10000;assert 'md5:'+hashlib.md5(raw).hexdigest()==item['checksum'];(OUT/'models_archive_description.md').write_bytes(raw);print(raw.decode('utf-8'),flush=True)
  (OUT/'metadata_fetch_status.json').write_text(json.dumps({'success':True,'official_url':url,'signal_or_model_archive_downloaded':False,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2),encoding='utf-8')
 except requests.RequestException as e:
  (OUT/'metadata_fetch_status.json').write_text(json.dumps({'success':False,'official_url':url,'error':repr(e),'signal_or_model_archive_downloaded':False,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2),encoding='utf-8');print('Metadata fetch failed',type(e).__name__,flush=True)
if __name__=='__main__':main()
