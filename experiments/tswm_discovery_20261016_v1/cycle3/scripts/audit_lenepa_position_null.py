"""Falsify an over-broad positional-collapse allegation using actual components.

Not full LeNEPA training or a paper reproduction. Source-default input targets
exclude absolute positional embeddings and both target and output are regularized.
The pure position-table null for an unrestricted embedding is not automatically
an admissible null of this implementation.
"""
import hashlib,importlib.util,json,pathlib
import numpy as np
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1]
SOURCE=ROOT/'sources'/'lenepa'
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def main():
 torch.set_num_threads(4);device='cuda';rows=[]
 source_pin=json.loads((SOURCE/'SOURCE_PIN.json').read_text())
 modules=module('lenepa_audit_modules',SOURCE/'models'/'modules.py');SIGReg=module('lenepa_audit_sigreg',SOURCE/'losses'/'sigreg.py').SIGReg
 config=(SOURCE/'configs'/'pretrain'/'objective'/'nepa_sigreg.yaml').read_text()
 assert 'nepa_target_uses_pos_embed: false' in config and 'nepa_target_layer: 0' in config and 'num_registers: 0' in config
 for seed in [41,42,43]:
  torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
  original=torch.randn(8,1,192,device=device).expand(-1,64,-1).clone()
  blocks=torch.nn.ModuleList([modules.Block(192,4,qkv_bias=False,bias=False,dropout=0.,attn_dropout=0.,use_rope=True,use_qk_norm=True,use_swiglu=True) for _ in range(8)]).to(device).eval()
  with torch.no_grad():
   h=original
   for block in blocks:h=block(h,is_causal=True)
   spread=(h-h[:,:1]).abs().max().item()
   assert spread<1e-4,'RoPE-only constant-input theorem failed numerically'
   loss=SIGReg(seed=seed).to(device)
   constant=torch.zeros(8,64,16,device=device)
   position=torch.randn(1,64,16,device=device).expand(8,-1,-1).clone()
   # Generic time-axis statistic is indeed blind to whether a Gaussian-looking
   # table is generated from content or absolute time, but the native target
   # tensor cannot be replaced by that table when the signal is constant.
   constant_loss=float(loss(constant));position_loss=float(loss(position))
   assert constant_loss>position_loss
  rows.append({'seed':seed,'native_rope_8blocks_constant_input_max_temporal_spread':spread,'generic_temporal_sigreg_constant_loss':constant_loss,'generic_temporal_sigreg_position_table_loss':position_loss,'generic_position_table_not_admissible_as_native_layer0_target':True})
 report={'source_revision':source_pin['revision'],'scope':'Unmodified native Block and SIGReg components, default no-absolute-PE/no-register objective configuration; no end-to-end data training','rows':rows,'source_facts':{'target_layer':0,'target_uses_absolute_position':False,'regularized_time_layers_published_recipe':[0,8],'objective_overrides_num_registers':0,'rope_qk_only':True},'decision':'REJECT_UNRESTRICTED_POSITION_TABLE_AS_PROOF_OF_NATIVE_LENEPA_FAILURE','remaining_unknown':'Does not prove the learned encoder preserves all content or solves long-horizon simulation. A different admissible failure needs independent evidence.'}
 out=ROOT/'outputs';(out/'lenepa_position_null_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
