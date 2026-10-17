"""Analytic G(reference)-G(positive); removes the shared score constant."""
import torch
from math57 import inv2,ld2

def relative_kernels(lam,u,reference,post):
    C,m=post['C'],post['m'];P=inv2(C);total=P+torch.diag(lam);W=inv2(total)
    h=(P@m[...,None]).squeeze(-1)
    linear=(W@h[...,None]).squeeze(-1)*lam
    quadratic=.5*(lam[:,None]*W*lam[None,:]-torch.diag(lam.square()/(1+lam)))
    difference=reference.permute(1,0,2)[None]-u[:,:,None,:]
    slope=linear+2*(quadratic@u[...,None]).squeeze(-1)
    H=(slope[:,:,None,:]*difference).sum(-1)+torch.einsum('btni,btij,btnj->btn',difference,quadratic,difference)
    diff=(u-m)*lam
    qf=.5*((C.diagonal(dim1=-2,dim2=-1)*lam).sum(-1)-ld2(total)-ld2(C)+torch.einsum('bti,btij,btj->bt',diff,W,diff))
    return H,qf
