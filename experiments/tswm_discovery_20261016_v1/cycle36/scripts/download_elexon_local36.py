"""Explicit local fallback after server HTTP403; normal TLS, public endpoints only."""
import pathlib,json,urllib.request,hashlib,datetime
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'public_metadata/elexon_raw_local'
OUT.mkdir(exist_ok=True);records=[]
for name,url in [
 ('schema.json','https://data.elexon.co.uk/swagger/v1/swagger.json'),
 ('windfor_20260101.json','https://data.elexon.co.uk/bmrs/api/v1/datasets/WINDFOR?publishDateTimeFrom=2026-01-01T00:00:00Z&publishDateTimeTo=2026-01-02T00:00:00Z&format=json'),
 ('fuelhh_wind_20260102.json','https://data.elexon.co.uk/bmrs/api/v1/datasets/FUELHH?settlementDateFrom=2026-01-02&settlementDateTo=2026-01-02&fuelType=WIND&format=json')]:
    req=urllib.request.Request(url,headers={'User-Agent':'TSWM-public-provenance-audit/1.0','Accept':'application/json'})
    with urllib.request.urlopen(req,timeout=40) as r:
        assert r.status==200;raw=r.read(12*1024*1024+1);assert len(raw)<=12*1024*1024
    json.loads(raw);(OUT/name).write_bytes(raw)
    records.append({'file':name,'url':url,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
                   'fetched_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'download_host':'local_client_after_server_403'})
    (OUT/'receipts.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(json.dumps({'file':name,'bytes':len(raw),'local_public_http':200}),flush=True)
