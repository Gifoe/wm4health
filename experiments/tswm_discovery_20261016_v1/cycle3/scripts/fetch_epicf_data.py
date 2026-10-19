"""Download public Zenodo v1.2 files with verified TLS and published checksums.

The workstation DNS gives 0.0.0.0 for zenodo.org. curl --resolve uses the public
DNS answer recorded below without changing system DNS, Host, SNI or TLS checks.
No credentials are used. The server is the intended execution location.
"""
import datetime,hashlib,json,pathlib,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[1]
RECORD=20762823
ADDRESS='188.185.40.132'  # Resolve-DnsName zenodo.org -Server 1.1.1.1, 2026-10-15 UTC
FILES=[('single_policy_data.csv',10071252,'14a8bec2987b67b17213e800f8516646'),
       ('multi_policy_data.csv',15581482,'c3c0fe0e0dcfc3aaf31caabc3e6ce1a7'),
       ('data_dictionary.csv',3821,'ddb366917875a2c970484340aec431d6')]
def digest(path,algorithm):
 h=hashlib.new(algorithm)
 with path.open('rb') as f:
  while b:=f.read(1024*1024):h.update(b)
 return h.hexdigest()
def main():
 dst=ROOT/'data'/'raw';dst.mkdir(parents=True,exist_ok=True);rows=[]
 for name,size,md5 in FILES:
  path=dst/name;url=f'https://zenodo.org/records/{RECORD}/files/{name}?download=1'
  if not path.exists():
   temporary=path.with_suffix(path.suffix+'.partial')
   subprocess.run(['curl.exe','--fail','--location','--silent','--show-error','--connect-timeout','20','--max-time','240','--retry','2','--resolve',f'zenodo.org:443:{ADDRESS}',url,'--output',str(temporary)],check=True)
   assert temporary.stat().st_size==size,(name,temporary.stat().st_size,size)
   assert digest(temporary,'md5')==md5,name
   temporary.replace(path)
  assert path.stat().st_size==size and digest(path,'md5')==md5,name
  row={'file':name,'bytes':size,'md5':md5,'sha256':digest(path,'sha256'),'url':url,'record':RECORD,'version':'1.2','license':'CC-BY-4.0','verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'transport':'curl HTTPS with normal certificate verification; explicit public DNS answer'}
  rows.append(row);print(json.dumps(row),flush=True)
 out=ROOT/'outputs';out.mkdir(exist_ok=True,parents=True);(out/'data_registry.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
if __name__=='__main__':main()
