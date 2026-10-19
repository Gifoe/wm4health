"""Fetch the original public benchmark archive, never execute/extract its contents."""
import urllib.request,datetime,pathlib,json,hashlib,zipfile,time,traceback
R=pathlib.Path(__file__).resolve().parents[1];O=R/'data_intake';O.mkdir(parents=True,exist_ok=True)
dest=O/'GEFCom2014_original.zip'
assert not dest.exists(),'Preserve existing raw archive; inspect receipt rather than redownload'
urls=['https://ars.els-cdn.com/content/image/1-s2.0-S0169207016000133-mmc1.zip',
      'https://dl.dropboxusercontent.com/s/pqenrr2mcvl0hk9/GEFCom2014.zip']
events=[]
for idx,url in enumerate(urls):
    tmp=O/f'fetch_attempt_{idx}.partial'
    assert not tmp.exists(),'Preserve previous attempt'
    event={'url':url,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'tswm-research-public-data-audit/1.0'})
        started=time.monotonic();h=hashlib.sha256();n=0
        with urllib.request.urlopen(req,timeout=45) as response, tmp.open('wb') as f:
            event.update(status=response.status,final_url=response.url,headers=dict(response.headers))
            while block:=response.read(1024*1024):
                n+=len(block);assert n<=256*1024*1024,'Unexpected archive size'
                f.write(block);h.update(block)
        assert zipfile.is_zipfile(tmp),'Public response is not a ZIP archive'
        with zipfile.ZipFile(tmp) as z: listing=[{'name':x.filename,'bytes':x.file_size,'compressed':x.compress_size,'crc32':x.CRC} for x in z.infolist()]
        tmp.replace(dest);event.update(success=True,bytes=n,sha256=h.hexdigest(),seconds=time.monotonic()-started,archive_listing=listing)
        events.append(event);(O/'download_receipts.json').write_text(json.dumps(events,indent=2),encoding='utf-8')
        print(json.dumps({'success':True,'bytes':n,'sha256':h.hexdigest(),'archive_entries':len(listing),'seconds':event['seconds']}),flush=True)
        break
    except Exception as e:
        event.update(success=False,error=repr(e));events.append(event)
        (O/'download_receipts.json').write_text(json.dumps(events,indent=2),encoding='utf-8')
        print(json.dumps({'success':False,'attempt':idx,'error':repr(e)}),flush=True)
else:raise RuntimeError('Both public archive transports failed; preserve receipts and use an explicit fallback')
