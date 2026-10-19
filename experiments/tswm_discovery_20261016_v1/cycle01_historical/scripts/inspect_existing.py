import json,pathlib,sys,torch
torch.set_num_threads(4)
ROOT=pathlib.Path(__file__).resolve().parents[1]
base=pathlib.Path('D:/baseline_reproduction_seed2021_v1')
out={}
for ds in ['NP','PJM']:
    for h in [24,360]:
        d=base/'results/worldts'/f'{ds}_H{h}_seed2021'
        ck=torch.load(d/'stages/stage2_decoder_mix.pt',map_location='cpu',weights_only=False)
        conf=json.loads((d/'resolved_config.json').read_text(encoding='utf-8'))
        out[f'{ds}_{h}']={'checkpoint_keys':list(ck)[:12],'config':conf,'checkpoint_bytes':(d/'stages/stage2_decoder_mix.pt').stat().st_size}
        print(ds,h,list(ck)[:12],json.dumps(conf),flush=True)
(ROOT/'outputs/existing_checkpoint_inspection.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
