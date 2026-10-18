"""Pin small official CSVs on serverG; inspect headers/dictionaries only."""
import sys,pathlib,urllib.request,urllib.parse,csv
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common72_v1 import *

def fetch(row,directory):
    key=row['key'];assert key in ('single_policy_data.csv','multi_policy_data.csv','data_dictionary.csv')
    url=row['links']['self'];p=urllib.parse.urlparse(url)
    assert p.scheme=='https' and p.hostname=='zenodo.org'
    directory.mkdir(exist_ok=True,parents=True);target=directory/key;partial=directory/(key+'.part')
    assert not target.exists() and not partial.exists()
    expected=row['size'];assert expected<20_000_000
    h=hashlib.sha256();md5=hashlib.md5();count=0
    req=urllib.request.Request(url,headers={'User-Agent':'research-provenance-audit/1.0'})
    with urllib.request.urlopen(req,timeout=45) as response,partial.open('xb') as f:
        final=response.geturl();assert urllib.parse.urlparse(final).hostname=='zenodo.org'
        while True:
            block=response.read(1_048_576)
            if not block:break
            count+=len(block);assert count<=expected
            h.update(block);md5.update(block);f.write(block)
    assert count==expected and row['checksum']=='md5:'+md5.hexdigest()
    partial.rename(target)
    with target.open(encoding='utf-8-sig',newline='') as f:header=next(csv.reader(f))
    return dict(file=key,storage_relative=str(target.relative_to(storage('data'))),bytes=count,
        sha256=h.hexdigest(),md5=md5.hexdigest(),source_url=url,header=header)

metadata=read(OUT/'METADATA_ACQUISITION72_V1.json')
assert metadata['status']=='PUBLIC_DATASET_VERSIONS_METADATA_CLOSED'
source=metadata['requested_record'];latest=metadata['latest_record']
assert source['id']==20680219 and source['version']=='1.1' and source['license']['id']=='cc-by-4.0'
same_signals={}
for name in ('single_policy_data.csv','multi_policy_data.csv'):
    a=next(r for r in source['files'] if r['key']==name);b=next(r for r in latest['files'] if r['key']==name)
    same_signals[name]=a['size']==b['size'] and a['checksum']==b['checksum']
assert all(same_signals.values()),'Different released signals require a new explicit version decision'
rows=[fetch(r,storage('data')/'zenodo20680219_v1_1') for r in source['files']]
extra=[]
if latest['id']!=source['id']:
    extra=[fetch(next(r for r in latest['files'] if r['key']=='data_dictionary.csv'),storage('data')/('documentation_record'+str(latest['id'])))]
result=dict(utc=utc(),status='OFFICIAL_SMALL_DATA_FILES_PINNED_HEADERS_ONLY',source_record=source['id'],source_version=source['version'],
    latest_record=latest['id'],latest_version=latest['version'],signal_payloads_identical_across_versions=same_signals,
    source_metadata_sha256=sha(OUT/'METADATA_ACQUISITION72_V1.json'),files=rows,additional_documentation=extra,
    data_license=source['license'],source_code_redistributed=False,counterfactual_targets_semantically_accessed=False,
    forecasts_computed=0,local_result_data_or_model_files_written=0)
dump(OUT/'DATA_ACQUISITION72_V1.json',result)
print(json.dumps(result),flush=True)
