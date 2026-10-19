"""Read only bounded public download headers; do not deserialize anything."""
import json,pathlib,requests,datetime
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 session=requests.Session();session.trust_env=False
 url='https://drive.usercontent.google.com/download?id=1ShZ_LyLGkWimboJv5RRk8M4yvrT7AUYY&export=download&confirm=t'
 row={'requested_url':url,'source':'Pinned official MTS3 data README','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 try:
  with session.get(url,stream=True,timeout=(8,30)) as response:
   row.update(status=response.status_code,content_type=response.headers.get('Content-Type'),content_length=response.headers.get('Content-Length'),content_disposition=response.headers.get('Content-Disposition'),resolved_url=response.url)
   prefix=next(response.iter_content(4096),b'');row.update(prefix_bytes_received=len(prefix),pickle_header=prefix[:2].hex(),html_response=(b'<html' in prefix.lower() or b'<!doctype' in prefix.lower()))
 except requests.RequestException as e:row['error']=type(e).__name__+': '+str(e)[:220]
 out=ROOT/'outputs/data_probe.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(row,indent=2),encoding='utf-8');print(json.dumps(row))
if __name__=='__main__':main()
