"""Export immutable pre-heldout evidence and a verified live-process snapshot."""
from common57 import *
import ctypes,math

def live(pid):
    fn=ctypes.windll.kernel32.OpenProcess;fn.argtypes=[ctypes.c_ulong,ctypes.c_bool,ctypes.c_ulong];fn.restype=ctypes.c_void_p
    handle=fn(0x1000,False,pid)
    assert handle,('Expected process absent',pid)
    try:
        value=ctypes.c_ulong();get=ctypes.windll.kernel32.GetExitCodeProcess;get.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
        assert get(handle,ctypes.byref(value)) and value.value==259,('Expected process not active',pid)
    finally:
        close=ctypes.windll.kernel32.CloseHandle;close.argtypes=[ctypes.c_void_p];close(handle)

def main():
    pins=source_check();state=read(OUT/'training_state57_v2.json');post=read(OUT/'post_state57_v2.json')
    assert state['status']=='RUNNING' and post['status']=='WAITING_ON_LIVE_TRAINING_OWNER'
    live(state['pid']);live(post['pid']);live(state['active']['pid'])
    assert not (OUT/'HELDOUT_OPEN_RECEIPT57_V1.json').exists()
    q=read(OUT/'QUALIFICATION57_V1.json');c=read(OUT/'CENTERED_QUALIFICATION57_V1.json');cuda=read(OUT/'CENTERED_CUDA_QUALIFICATION57_V1.json')
    assert q['math_sha256']==sha(ROOT/'scripts/math57.py')
    assert c['relative_math_sha256']==cuda['relative_math_sha256']==sha(ROOT/'scripts/relative57_v1.py')
    progress={'utc':utc(),'status':'C57_QUALIFIED_MATCHED_TRAINING_LIVE_HELDOUT_UNOPENED','training_owner_pid':state['pid'],'active_training_child_pid':state['active']['pid'],'active_tag':state['active']['tag'],'postprocess_owner_pid':post['pid'],'all_three_process_handles_verified_live':True,'validated_terminal_fits_at_snapshot':len(state['completed']),'planned_valid_unique_fits':32,'retained_failed_updates':10000,'heldout_open_receipt_absent':True,'protocol_sha256':sha(ROOT/'PROTOCOL.md'),'training_plan_sha256':sha(OUT/'TRAINING_PLAN57_V2.json'),'decision_rule_sha256':sha(ROOT/'DECISION_RULE_V1.md'),'source_files_sha256':pins,'overall_goal':'ACTIVE_UNACHIEVED','accepted_final_directions':0,'local_result_files_written':0,'scope':'Timestamped in-progress receipt, not a final result or continuing-liveness guarantee.'}
    dump(OUT/'PROGRESS_RECEIPT57_V1.json',progress)
    candidates=sorted(ROOT.glob('*.md'))+sorted(ROOT.glob('*.ps1'))+sorted((ROOT/'scripts').glob('*.py'))
    names=['NATIVE_FIXTURE57_V1.json','QUALIFICATION57_V1.json','DATA57_V1.json','FAILED_ATTEMPT57_V1.json','CENTERED_QUALIFICATION57_V1.json','CENTERED_CUDA_QUALIFICATION57_V1.json','TRAINING_PLAN57_V1.json','TRAINING_PLAN57_V2.json','PROGRESS_RECEIPT57_V1.json','preflight_state57_v1.json','training_state57_v1.json','post_state57_v1.json','native_fixture57_v1.py.stdout.log','native_fixture57_v1.py.stderr.log','qualify57_v1.py.stdout.log','qualify57_v1.py.stderr.log','generate57_v1.py.stdout.log','generate57_v1.py.stderr.log','running09_linear_seed2021.stdout.log','running09_linear_seed2021.stderr.log','runs/running09_linear_seed2021/result.json','runs/running09_linear_seed2021/curve.jsonl']
    candidates += [OUT/n for n in names]
    assert len(candidates)==len(set(candidates))
    dest=OUT/'publication_progress_v1';dest.mkdir(exist_ok=False);rows=[]
    for p in sorted(candidates):
        original=p.read_bytes();text=original.decode('utf-8-sig');data=text.encode('utf-8');public=p.relative_to(ROOT).as_posix()
        assert len(data)<1000000,(public,len(data))
        target=dest/'files'/public;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:stream.write(data)
        rows.append({'path':public,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'original_sha256':hashlib.sha256(original).hexdigest(),'git_blob_sha1':hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest(),'parts':math.ceil(len(text)/3500),'server_snapshot_file':str(target),'publication_transform':'Remove UTF8 BOM' if original.startswith(b'\xef\xbb\xbf') else None})
    manifest={'utc':utc(),'status':'C57_PRE_HELDOUT_QUALIFIED_PROGRESS_PUBLICATION','files':rows,'file_count':len(rows),'bytes':sum(x['bytes'] for x in rows),'total_content_parts':sum(x['parts'] for x in rows),'progress_receipt_sha256':sha(OUT/'PROGRESS_RECEIPT57_V1.json'),'scope':'Executable protocols, exact-source/CUDA qualifications, two explicit training plans, scorer guards and retained failed unshifted adaptation logs. The retained partial result.json is intentionally malformed and excluded from valid model evidence. No heldout result or final-direction claim. Large data/checkpoints remain on server.','overall_goal':'ACTIVE_UNACHIEVED','accepted_final_directions':0,'local_result_files_written':0}
    dump(dest/'manifest.json',manifest)
    print(json.dumps({'status':manifest['status'],'files':len(rows),'bytes':manifest['bytes'],'parts':manifest['total_content_parts'],'manifest_sha256':sha(dest/'manifest.json'),'progress_receipt_sha256':sha(OUT/'PROGRESS_RECEIPT57_V1.json')}),flush=True)

if __name__=='__main__':main()
