import json,pathlib,requests
ROOT=pathlib.Path(__file__).resolve().parents[1]
s=requests.Session();s.trust_env=False
with s.get('https://keeper.mpdl.mpg.de/f/988ed8adf071486c80dc/?dl=1',headers={'Range':'bytes=0-0'},stream=True,timeout=(20,25)) as r:
    row=dict(status=r.status_code,content_range=r.headers.get('Content-Range'),content_length=r.headers.get('Content-Length'),accept_ranges=r.headers.get('Accept-Ranges'))
    if r.status_code==206:row['byte']=next(r.iter_content(1)).hex()
(ROOT/'outputs/external/explore_context/range_support.json').write_text(json.dumps(row,indent=2),encoding='utf-8');print(json.dumps(row))
