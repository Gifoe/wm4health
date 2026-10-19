"""Wait for the existing job; never launch or restart scientific training."""
import pathlib,json,hashlib,subprocess,sys,time,datetime,os,traceback
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(value):
    p=OUT/'postprocess_status.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');tmp.replace(p)
def main():
    assert not (OUT/'postprocess_status.json').exists()
    steps=['evaluate_readouts33.py','verify_readouts33_v2.py','analyze_readouts33.py','close_provenance33.py','plot_readouts33.py']
    locks={s:sha(ROOT/'scripts'/s) for s in steps}
    record={'status':'waiting_for_existing_training','pid':os.getpid(),'started_utc':stamp(),
        'script_hashes':locks,'never_restarts_training':True,'steps':[]}
    write(record);start=time.monotonic()
    while True:
        supervisor=json.loads((OUT/'supervisor.json').read_text())
        if supervisor['status']=='failed':raise RuntimeError('Existing training failed; inspect preserved logs')
        if supervisor['status']=='completed':
            assert supervisor['exit_code']==0
            seal=json.loads((OUT/'readout_fits_v1/training_complete.json').read_text())
            assert seal['status']=='complete_all_24_fits_test_not_loaded' and len(seal['fits'])==24
            break
        if time.monotonic()-start>7200:raise TimeoutError('No training completion; no duplicate launched')
        time.sleep(5)
    record['training_seal_sha256']=sha(OUT/'readout_fits_v1/training_complete.json')
    for s in steps:
        assert sha(ROOT/'scripts'/s)==locks[s]
        record.update(status='running',current_step=s);write(record)
        with (OUT/(s+'.stdout.log')).open('xb') as out,(OUT/(s+'.stderr.log')).open('xb') as err:
            p=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'scripts'/s)],cwd=str(ROOT),stdout=out,stderr=err,
                creationflags=subprocess.CREATE_NO_WINDOW)
            code=p.wait()
        record['steps'].append({'script':s,'exit_code':code,'ended_utc':stamp()});write(record)
        if code:raise RuntimeError(f'{s} failed with exit {code}; outputs preserved')
    record.update(status='completed',ended_utc=stamp());write(record)
if __name__=='__main__':
    try:main()
    except Exception:
        p=OUT/'postprocess_status.json'
        rec=json.loads(p.read_text()) if p.exists() else {'pid':os.getpid()}
        rec.update(status='failed',traceback=traceback.format_exc(),ended_utc=stamp());write(rec)
        raise
