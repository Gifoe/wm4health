# Model, data and execution registry

| Component | Definition | Interpretation |
|---|---|---|
| Native CSDI, random | Original CSDI_base, native random masks, full Transformer attention | Existing observational conditional generator; source algorithms unchanged. |
| Native CSDI, coverage | Same model and optimizer, .5 random/.25 block/.25 prefix training mask | Ordinary query-distribution control, not a proposed contribution. |
| Frozen block query | Reveal all planned U and no Y | Observational conditional; not automatically a do law. |
| Frozen prefix query | At each t reveal only generated Y_<t and U_<=t; keep current Y | Classical sequential g-computation through the identical frozen checkpoint. |
| Causal MDN | Conventional two-input width224/224 MLP, three Gaussian mixture components | Existing Markov transition-density baseline. |
| Causal AR | Train-only linear Gaussian regression | Correctly specified on linear system; misspecified nonlinear control. |
| Exact iid references |512 samples of do or exact observational conditional law | Privileged finite-sampling floor, no fitting. |
| Bayes DSM and native sampler references | Analytic linear Gaussian conditioning and original DDPM sampler | Privileged attribution, not learned methods. |

CSDI has51377 parameters; causal MDN has53097. Same records and fixed optimizer-update budgets do not imply identical FLOPs or information constraints. MDN has the correct Markov factorization and uses only Y transition targets; CSDI learns joint Y/U conditionals. Target-scalar exposures and measured per-fit runtimes are preserved in FITS65_V1.json. No claim of matched compute or superior new architecture is made.

Six independent corpora: linear/nonlinear mechanisms x seeds2021/2022/2023. Each contains16384 training,4096 validation and8192 reserved observational-test trajectories of length8, with explicit array channel0=Y, channel1=U. Known initial state=.25; behavior U=-.75 previousY+.5eta. Innovations are independent, all states observed, and the continuous Gaussian behavior policy has full support. No hidden confounding or clinical semantics is asserted.

Training: CSDI channels32/layers2/heads4/diffusion embedding32/time16/feature8; native50-step quadratic .0001-to-.5 schedule, Adam.001, batch64. MDN Adam.001, batch512 transitions. Fixed checkpoints2000/8000; common block/prefix validation DSM chooses CSDI, one-step validation NLL chooses MDN. Test truth never selects settings or models. Native source commit7f24a436f08d98853a6b43d4f7f04e5a65ecdf27 with MIT license is retained.

Final successful formal updates=144000; original discarded model updates >=8000, plus >=4000 duplicated recovery updates and qualification fixtures listed separately. The original12 successful fits use the ordinary startup stack; the six recovery fits use isolated execution with the same PyTorch2.8.0+cu128. Final two fits use numeric NPZ/JSON checkpoints; stored model/optimizer/RNG values are unchanged by the format. All scored models and validation checks are replayed under the common isolated runtime. NumPy source-path difference and failed native execution are explicit reproducibility limitations, not silently omitted.
