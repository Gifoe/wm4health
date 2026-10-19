"""Read-only HEAD requests to HTTPS S3 aliases; never disables TLS verification."""
import json,requests
urls=['https://causalchamber.s3.eu-central-1.amazonaws.com/downloadables/wt_walks_v1.zip','https://s3.eu-central-1.amazonaws.com/causalchamber/downloadables/wt_walks_v1.zip','https://causalchamber.s3.amazonaws.com/downloadables/wt_walks_v1.zip']
s=requests.Session();s.trust_env=False
for url in urls:
 try:
  r=s.head(url,timeout=(7,15),allow_redirects=False);print(json.dumps({'url':url,'status':r.status_code,'bytes':r.headers.get('Content-Length'),'redirect':r.headers.get('Location')}),flush=True)
 except requests.RequestException as e:print(json.dumps({'url':url,'error':type(e).__name__}),flush=True)
