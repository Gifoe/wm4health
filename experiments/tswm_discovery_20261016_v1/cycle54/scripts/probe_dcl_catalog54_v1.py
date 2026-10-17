"""Bounded public metadata-only requests from the server; never download signals."""
import concurrent.futures, datetime, hashlib, json, pathlib, time, urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[1]
URLS=['https://nefeli.helmholtz-munich.de/records/tdfs2-kx054','https://nefeli.helmholtz-munich.de/api/records/tdfs2-kx054']
LIMIT=512*1024

def probe(url):
    start=time.perf_counter()
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'TSWMResearchMetadataAudit/1.0'})
        with urllib.request.urlopen(req,timeout=18) as response:
            raw=response.read(LIMIT+1); assert len(raw)<=LIMIT,'Metadata exceeds bounded read; no further data downloaded'
            result={'url':url,'final_url':response.url,'status':response.status,'content_type':response.headers.get('Content-Type'),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
            if 'json' in result['content_type']:
                record=json.loads(raw)
                result['record_id']=record.get('id'); result['metadata']=record.get('metadata')
                result['files']=record.get('files'); result['links']=record.get('links')
            else:
                text=raw.decode('utf-8',errors='replace')
                result['contains_record_id']='tdfs2-kx054' in text
            result['elapsed_seconds']=time.perf_counter()-start
            return result
    except Exception as exc:
        return {'url':url,'error_type':type(exc).__name__,'error':str(exc),'elapsed_seconds':time.perf_counter()-start}

def main():
    output=ROOT/'outputs/dcl_catalog_probe54_v1.json'; assert not output.exists()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool: records=list(pool.map(probe,URLS))
    obj={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'METADATA_AVAILABILITY_PROBE_COMPLETE','requests':records,'read_limit_bytes_per_request':LIMIT,'signals_downloaded':False,'formal_training_changed':False,'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
    output.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8'); print(json.dumps(obj),flush=True)
if __name__=='__main__': main()
