# Exact observed-law witness and its limit

This is a controlled structural example, not a new identification theorem or another sample from the unchanged released DoTime prior.

Let x=(U,A,Y), with observed projection C selecting A,Y. The structural equation is (I-B)x[t]=D x[t-1]+epsilon[t], epsilon iid N(0,0.09I). Set a=1,b=3/5,c=-1 in the triangular instantaneous matrix B, with B[A,U]=a, B[Y,A]=b, B[Y,U]=c. Let D=s diag(1/5,1/2,7/10).

Define the hidden-coordinate-only transform

```
S = [[1, -1/15, 2/3],
     [0,  1,     0  ],
     [0,  0,     1  ]].
```

Alternative B' has a'=1,b'=-2/5,c'=1. Put L=(I-B)^-1, L'=(I-B')^-1, F=LD, F'=S F S^-1 and D'=L'^-1 F'. The matrix O=L'^-1 S L is

```
O = [[ 2/3, 1/3,  2/3],
     [ 1/3, 2/3, -2/3],
     [-2/3, 2/3,  1/3]].
```

The exact Fraction-arithmetic verifier checks OO^T=I and CS^-1=C. Thus epsilon'=O epsilon is still independent equal-variance Gaussian structural noise. With x'=Sx, the alternative structural system has exactly the same complete observed process. This is equality of process laws, not agreement of a finite list of moments. The direct coupling is one way to prove that equality; the two worlds need not share an unobserved physical noise realization in deployment.

A hard intervention replacing A[t] by its factual value+1 changes Y[t] by b=+3/5 in the first world and b'=-2/5 in the second. The retrospective factual-value offset defines a local response audit. It supplies no future observation to a prospective predictor.

At positive s, D' generally has cross-lags, while D is diagonal. Therefore this is a statement about relaxing the lag-mask class, not a disproof of positive-lag identification within the original class. Both transition matrices have the same stable eigenvalues. At s=0 both reduce to the iid boundary.

The norm of D' off the diagonal equals approximately0.749837842552*s. Making it arbitrarily small also makes all temporal persistence arbitrarily weak. The example does not prove that tiny relative perturbations overturn an otherwise strongly identified persistent process. This limitation is part of the result.

Fifteen unmodified native-simulator checks, direct float64 recurrences and independent spectral calculations support the exact construction. Native float32 mechanism parameters cause the documented sub-micro-unit discrepancies. The rational proof and raw arrays are retained. Ordinary randomized do experiments recover the slope without any new architecture; this example supplies no demonstrated new-method gap.
