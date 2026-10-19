"""Run the disclosed complete PEMS03 configuration, with durable raw logs."""
import datetime,hashlib,json,os,pathlib,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
def dump(p,o):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,indent=2),encoding='utf-8')
def main():
 tag=os.environ.get('TSWM_KUMA_OUT','kuma_native_pems03');assert '/' not in tag and '\\' not in tag and tag not in ['.','..']
 out=ROOT/'outputs'/tag;out.mkdir(parents=True,exist_ok=True)
 assert (ROOT/'data'/'PEMS03.npz').exists()
 args=[sys.executable,'-B','-u',str(ROOT/'sources'/'kuma'/'run.py'),'--mode','train','--root_path',str(ROOT/'data'),'--data_path','PEMS03.npz','--enc_in','358','--activation','gelu','--gpu','0','--random_seed','2025','--d_model','256','--d_ff','256','--e_layers','1','--batch_size','32','--num_workers','0','--train_epochs','20','--learning_rate','0.002','--output_dir',str(out)]
 plan={'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Disclosed full native KUMA PEMS03 96->12 task, model/loss/optimizer/dataloader/evaluation unmodified. Not a short demo.','revision':'8ed21081366309e313f821f3dc760f6ac945ac4b','data_sha256':hashlib.sha256((ROOT/'data'/'PEMS03.npz').read_bytes()).hexdigest(),'argv':args,'runtime_difference':'num_workers0 instead of launcher10 for Windows; no training/eval batch caps; published early stopping and LR schedule unchanged','checkpoint':'Native validation loss; test only after native training selection','goal':'Healthy source-faithful reproduction before new mechanism attribution, not a new research idea'}
 plan['runtime_profile']={'threads':os.environ.get('TSWM_CPU_THREADS','4'),'affinity':'0xffff0000' if 'affinity1' in tag else 'default','mkl_threading':os.environ.get('MKL_THREADING_LAYER','default')}
 if (out/'plan.json').exists():raise RuntimeError('Existing native run; inspect its live process/results instead of starting a duplicate')
 dump(out/'plan.json',plan)
 env=os.environ.copy();threads=os.environ.get('TSWM_CPU_THREADS','4');env.update(OMP_NUM_THREADS=threads,MKL_NUM_THREADS=threads,PYTHONHASHSEED='2025')
 with (out/'stdout.log').open('w',encoding='utf-8') as log:
  proc=subprocess.Popen(args,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
  dump(out/'process.json',{'pid':proc.pid,'executable':sys.executable,'argv':args,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
  last=''
  while proc.poll() is None:
   time.sleep(5)
   lines=(out/'stdout.log').read_text(encoding='utf-8',errors='replace').splitlines()
   tail=lines[-1] if lines else 'Native run starting'
   if tail!=last:print(tail,flush=True);last=tail
  code=proc.wait()
 dump(out/'status.json',{'state':'complete' if code==0 else 'failed','exit_code':code,'process_id':proc.pid,'scope':'Single full native configuration, seed2025','objective_achieved':False})
 print((out/'stdout.log').read_text(encoding='utf-8',errors='replace')[-5000:],flush=True)
 if code:raise SystemExit(code)
if __name__=='__main__':main()
