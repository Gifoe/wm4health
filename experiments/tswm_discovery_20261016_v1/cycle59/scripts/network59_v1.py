"""Inspect transport reachability without revealing proxy credentials or changing settings."""
from common59 import *
import urllib.request,urllib.parse,time
proxies={k:{'scheme':urllib.parse.urlparse(v).scheme,'host':urllib.parse.urlparse(v).hostname,'port':urllib.parse.urlparse(v).port} for k,v in urllib.request.getproxies().items()}
results=[]
urls=['https://raw.githubusercontent.com/minjiechen/magnetchallenge-2/ab4913ca63ebda1f8eab429c64d3e93f96204713/tutorials/tutorial-2/README.md','https://api.github.com/repos/minjiechen/magnetchallenge-2','https://cdn.jsdelivr.net/gh/minjiechen/magnetchallenge-2@ab4913ca63ebda1f8eab429c64d3e93f96204713/tutorials/tutorial-2/README.md']
for mode in ['default','direct']:
    opener=urllib.request.build_opener() if mode=='default' else urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for url in urls:
        start=time.monotonic()
        try:
            with opener.open(urllib.request.Request(url,headers={'User-Agent':'tswm-data-audit'}),timeout=4) as r:
                sample=r.read(64);row={'mode':mode,'host':urllib.parse.urlparse(url).hostname,'status':r.status,'read_bytes':len(sample)}
        except Exception as e:row={'mode':mode,'host':urllib.parse.urlparse(url).hostname,'error_type':type(e).__name__,'error':str(e)}
        row['seconds']=time.monotonic()-start;results.append(row);print(json.dumps(row),flush=True)
obj={'utc':utc(),'status':'READ_ONLY_TRANSPORT_PROBE','sanitized_proxy_endpoints':proxies,'requests':results,'settings_changed':False}
dump(OUT/'NETWORK59_V1.json',obj)
