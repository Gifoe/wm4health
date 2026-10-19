"""Download the small CC-BY4.0 UCI dataset, preserving exact archive and hashes."""
import datetime, hashlib, json, pathlib, zipfile
import requests

ROOT=pathlib.Path(__file__).resolve().parents[1]
URL='https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip'

def main():
    data=ROOT/'data';data.mkdir(exist_ok=True);dest=data/'bike_sharing.zip'
    if not dest.exists():
        s=requests.Session();s.trust_env=False;r=s.get(URL,timeout=(5,30));r.raise_for_status();assert len(r.content)<5_000_000;dest.write_bytes(r.content)
    with zipfile.ZipFile(dest) as z:
        for name in ['hour.csv','Readme.txt']:
            item=next(i for i in z.infolist() if pathlib.PurePosixPath(i.filename).name.lower()==name.lower())
            assert item.file_size<5_000_000;(data/name).write_bytes(z.read(item))
    rp=ROOT/'outputs/data_registry.json';registry=json.loads(rp.read_text(encoding='utf-8'))
    assert not any(r['key']=='BikeSharing' for r in registry['datasets'])
    raw=(data/'hour.csv').read_bytes();registry['datasets'].append({'key':'BikeSharing','path':'data/hour.csv','source':URL,
        'source_page':'https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset','license':'CC BY4.0','citation':'Fanaee-T,H.(2013),Bike Sharing,UCI,DOI10.24432/C5W894',
        'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'archive_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
        'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    rp.write_text(json.dumps(registry,indent=2),encoding='utf-8');print(json.dumps({'dataset':'BikeSharing','bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}))

if __name__=='__main__':main()
