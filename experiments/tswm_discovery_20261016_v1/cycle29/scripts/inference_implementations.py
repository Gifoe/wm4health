"""Existing algebraic GRU folding and source-attention SDPA controls; no new method."""
import types
import torch
from torch import nn
from torch.nn import functional as F


@torch.no_grad()
def prepare_fused_gru(model, encoded):
    h, condition = encoded
    assert len(h) == 1, 'one physical system per fused shared-weight plan batch'
    gru = nn.GRU(1, model.width, batch_first=True).to(h.device, h.dtype).eval()
    gru.weight_ih_l0.copy_(model.cell.weight_ih[:, :1])
    gru.weight_hh_l0.copy_(model.cell.weight_hh)
    gru.bias_ih_l0.copy_(model.cell.bias_ih)
    gru.bias_hh_l0.copy_(model.cell.bias_hh)
    if model.kind == 'compiled64':
        gru.weight_hh_l0.add_(condition[0])
    else:
        gru.bias_ih_l0.add_(F.linear(condition, model.cell.weight_ih[:, 1:])[0])
    gru.flatten_parameters()
    return gru, h


@torch.no_grad()
def fused_simulate(model, prepared, future_u):
    gru, h = prepared
    states, _ = gru(future_u, h.unsqueeze(0).expand(1, len(future_u), -1).contiguous())
    return model.readout(states)


def _self_attention(self, x):
    if self.causal:
        mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1], device=x.device)
        value = self.mha(x, x, x, attn_mask=mask, is_causal=True, need_weights=False)[0]
    else:
        value = self.mha(x, x, x, is_causal=False, need_weights=False)[0]
    return self.resid_dropout(value)


def _cross_attention(self, x, mem):
    value = self.mha(x, mem, mem, is_causal=self.causal, need_weights=False)[0]
    return self.resid_dropout(value)


def enable_unused_attention_weight_elision(model):
    """Only mutate a separately constructed inference copy, never original source."""
    from transformer_sim import SelfAttention, CrossAttention
    count = 0
    for module in model.modules():
        if isinstance(module, SelfAttention):
            module.forward = types.MethodType(_self_attention, module)
            count += 1
        elif isinstance(module, CrossAttention):
            module.forward = types.MethodType(_cross_attention, module)
            count += 1
    assert count > 0
    return count
