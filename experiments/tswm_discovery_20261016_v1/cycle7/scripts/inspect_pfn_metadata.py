"""Read-only metadata screening; do not download the complete11GB checkpoint repo."""
import datetime,json,pathlib
import requests
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 errors=[]
 for endpoint in ['https://hf-mirror.com','https://huggingface.co']:
  session=requests.Session();session.trust_env=False
  try:
   response=session.get(endpoint+'/api/models/thummd/do-over-time-pfn?blobs=true',timeout=(8,15));response.raise_for_status();data=response.json()
   selected=[x for x in data.get('siblings',[]) if x['rfilename'].startswith('s9ho_all_causal/')]
   result={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'endpoint':endpoint,'revision':data['sha'],'repository':'thummd/do-over-time-pfn','selected_metadata':selected,'all_checkpoint_count':sum(x['rfilename'].endswith('.pt') for x in data.get('siblings',[])),'downloaded_weights':False,'errors':errors}
   (ROOT/'outputs/dotime_pfn_metadata.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result));return
  except (requests.RequestException,ValueError,KeyError) as error:errors.append({'endpoint':endpoint,'error':repr(error)})
 (ROOT/'outputs/dotime_pfn_metadata_unavailable.json').write_text(json.dumps({'errors':errors,'downloaded_weights':False},indent=2),encoding='utf-8');print(json.dumps({'errors':errors,'downloaded_weights':False}))
if __name__=='__main__':main()
