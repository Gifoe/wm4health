"""Acquire only public dataset metadata and a data dictionary, no signal/target CSVs."""
import sys,pathlib,urllib.request,urllib.parse,re
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common72_v1 import *

def get(url):
    parsed=urllib.parse.urlparse(url)
    assert parsed.scheme=='https' and parsed.hostname=='zenodo.org'
    req=urllib.request.Request(url,headers={'User-Agent':'research-provenance-audit/1.0','Accept':'application/json'})
    with urllib.request.urlopen(req,timeout=30) as response:
        body=response.read(3_000_001)
        assert len(body)<=3_000_000,'Metadata limit exceeded'
        return body,response.geturl()

def summary(obj):
    return dict(id=obj['id'],version=obj.get('metadata',{}).get('version'),
        title=obj.get('metadata',{}).get('title'),license=obj.get('metadata',{}).get('license'),
        access_right=obj.get('metadata',{}).get('access_right'),links=obj.get('links'),
        files=[dict(key=f.get('key'),size=f.get('size'),checksum=f.get('checksum'),links=f.get('links')) for f in obj.get('files',[])])

result=dict(utc=utc(),status='PUBLIC_METADATA_ACQUISITION_STARTED',requests=[],errors=[],raw_target_data_downloaded=False,counterfactual_targets_semantically_accessed=False)
try:
    body,url=get('https://zenodo.org/api/records/20680219');record=json.loads(body)
    dump(OUT/'ZENODO_RECORD20680219_V1.json',record)
    result['requested_record']=summary(record)
    result['requests'].append(dict(url=url,bytes=len(body),sha256=hashlib.sha256(body).hexdigest()))
    latest_url=record.get('links',{}).get('latest')
    if latest_url and latest_url!='https://zenodo.org/api/records/20680219':
        body,url=get(latest_url);latest=json.loads(body)
        dump(OUT/'ZENODO_LATEST_METADATA72_V1.json',latest)
        result['latest_record']=summary(latest)
        result['requests'].append(dict(url=url,bytes=len(body),sha256=hashlib.sha256(body).hexdigest()))
    else:result['latest_record']=result['requested_record']
    result['status']='PUBLIC_DATASET_VERSIONS_METADATA_CLOSED'
except Exception as e:
    result['status']='PUBLIC_METADATA_REQUEST_FAILED_WITHOUT_DATA_DOWNLOAD'
    result['errors'].append(dict(type=type(e).__name__,message=str(e)))
dump(OUT/'METADATA_ACQUISITION72_V1.json',result)
print(json.dumps(result),flush=True)
