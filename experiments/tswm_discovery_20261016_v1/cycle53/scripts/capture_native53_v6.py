"""Resume only missing fits; validate native terminal state, never just exit code."""
from native53 import *
import subprocess

state=R/'outputs/supervisor_v6.json';assert not state.exists()
assert json.loads((R/'outputs/microbatch_qualification_v2/qualification.json').read_text())['status']=='PASS_NATIVE_EFFECTIVE_BATCH_MICROBATCHING'
assert (R/'outputs/RESOURCE_ABORT_RECEIPT_V1.json').exists()
assert (R/'outputs/SUPERVISOR_RESOURCE_REPAIR_RECEIPT_V1.json').exists()
plan=json.loads((R/'outputs/FORMAL_JOB_PLAN_V1.json').read_text());assert plan['protocol_sha256']==sha(R/'PROTOCOL.md')
completed=[];selections=[]
def save(status,**extra):dump(state,{'status':status,'pid':os.getpid(),'utc':now(),'completed':completed,**extra})
def valid(folder,tag):
    if not (folder/'state.json').exists() or not (folder/'summary.json').exists():return None
    terminal=json.loads((folder/'state.json').read_text())
    if terminal.get('status')!='COMPLETE' or terminal.get('tag')!=tag:return None
    summary=json.loads((folder/'summary.json').read_text())
    if not pathlib.Path(summary['checkpoint']).is_file() or sha(summary['checkpoint'])!=summary['checkpoint_sha256']:return None
    return summary
env=os.environ.copy();env['PYTHONPATH']=str(P/'cycle52/runtime_deps')+os.pathsep+env.get('PYTHONPATH','');env['WANDB_MODE']='disabled'
save('RUNNING',repair='effective batch128 /microbatch64 for memory-saturated PatchTST concat only; incomplete V1 attempt retained and excluded')
for job in plan['jobs']:
    tag=job['tag'];prior=valid(ART/tag,tag)
    if prior is not None:
        completed.append({'tag':tag,'reused_terminal_v1':True,'checkpoint_sha256':prior['checkpoint_sha256'],'summary_path':str(ART/tag/'summary.json')})
        if tag!='shared_codec':selections.append({'tag':tag,'checkpoint':prior['checkpoint'],'sha256':prior['checkpoint_sha256'],'best_epoch':prior['best_epoch'],'summary_path':str(ART/tag/'summary.json')})
        save('RUNNING');continue
    assert tag!='shared_codec','Codec is not terminal; inspect before attempting another preparation'
    folder=ART/'runtime_microbatch_v6'/tag
    assert not folder.exists(), 'Existing V2 work; inspect instead of restarting'
    log=R/'outputs'/('native_v6_'+tag+'.log')
    with log.open('wb') as stream:
        child=subprocess.Popen([sys.executable,'-B','-u',str(R/'scripts/train_native53_v6.py'),*job['arguments']],cwd=R,env=env,stdout=stream,stderr=subprocess.STDOUT)
        save('RUNNING',active={'tag':tag,'pid':child.pid,'log':log.name})
        code=child.wait()
    summary=valid(folder,tag)
    if code or summary is None:
        save('FAILED_NATIVE_JOB',error=tag,exit_code=code,native_terminal_files_valid=summary is not None);sys.exit(1)
    completed.append({'tag':tag,'pid':child.pid,'exit_code':code,'reused_terminal_v1':False,'summary_path':str(folder/'summary.json')})
    selections.append({'tag':tag,'checkpoint':summary['checkpoint'],'sha256':summary['checkpoint_sha256'],'best_epoch':summary['best_epoch'],'summary_path':str(folder/'summary.json')})
    save('RUNNING')
assert len(completed)==41 and len(selections)==40
assert len(set(s['tag'] for s in selections))==40
dump(R/'outputs/SELECTION_SEAL_V6.json',{'utc':now(),'protocol_sha256':sha(R/'PROTOCOL.md'),'selected_checkpoints':selections,'second_batch_evaluation_opened':False,'runtime_repair':'Effective batch128 /microbatch64 for memory-saturated PatchTST concat; iid dropout draw ordering differs; original terminal fits reused; incomplete native attempt excluded'})
save('ALL_40_NATIVE_FITS_COMPLETE_TRANSFER_UNOPENED')
