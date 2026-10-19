# Frozen protocol: coarse-time forcing interface falsifier

Frozen before cycle26 scientific execution on 2026-10-16. This is a new controlled mechanism audit, not a native-paper reproduction or a proposed new method. It tests whether perfect autonomous transition knowledge can still give a poor forced response solely because an input is inserted at the wrong time. Broad response/memory problems already have direct prior work.

## Source and task

Falasca, PRResearch7,043314 (2025), exact manuscript 2506.22552v4, partially observed triad parameters: L11=-2,L12=.2,L13=.1,gamma2=.6,gamma3=.4,sigma2=1.2,sigma3=.8,epsilon=.01. Source Euler-Maruyama dt=.01 is the reference **discrete process**, not an exact continuous-SDE solution. Its drift is (-2*x1+.2*x2+.1*x3+I*x1*x2, -.2*x1-I*x1**2-60*x2, -.1*x1-40*x3); noise=(0,12,8). I=1 is source nonlinear case; I=0 is an explicitly modified linear positive control. The audit exposes all three states to eliminate state estimation, representation loss and insufficient model capacity as explanations. It makes no claim to solve the source's partial-observation learning problem.

Falasca-Zanna2602.13847v5 native released response code adds dt_phys*force after each learned map. At its native one-Euler-step interval that is consistent with the discrete source. Here we deliberately test **an extension to coarser maps**, not accuse the original fine-step experiment of a bug. Repository pinned398082d6dcee2e9d42832dfcd39febfa48c383dd. Source files are separately archived with Git blob checks.

## Design, all retained

- Three independent Monte Carlo cohorts seeds20262601/2/3,8192 initial states each, burn10 physical time units from zero under unforced source process. Same initial states and subsequent fine Gaussian increments across all methods, controls and cadences within a source case. I=0 and1 have separately burned initial states using the same random streams.
- Native fine step=.01, rollout6.4 time units, coarse stride K=1,4,16,64. Observe common times0,.64,...,6.4. Five constant forcing branches u=0,+/-.01,+/-.1 acting on physical coordinate1 throughout every fine interval.
- Reference: add dt*u to coordinate1 every fine Euler step.
- Endpoint: exact unforced K-step map, then add K*dt*u to coordinate1.
- Symmetric: half K*dt*u before and half after the exact unforced K-step map. At K=1 use endpoint to retain the source-native definition; explicitly label this convention.
- Split4: four submaps with endpoint forcing for K>=4; at K=1 use endpoint. Same total fine dynamics calls and random increments, more frequent input insertions. This is ordinary subcycling, not a new model. At K=4 it equals the reference.
- Linear-only correct coarse input: exact discrete ZOH operator G_K=dt*sum_{j=0}^{K-1}(Id+dt*A)^j*e1 inserted after each exact passive map. This is an old control baseline. No fitted parameters or test-selected budgets.
- All autonomous paths must match reference exactly (roundoff permitted). The continuous-time input integral must not be substituted for this discrete-process operator.

## Outcomes and decision

Primary: normalized RMS difference between approximate and true paired forced-minus-unforced paths across common times and physical systems, coordinate1 and all3 separately. Secondary: relative error of population mean response over time, endpoint mean response, variance response and factual autonomous path discrepancy. Report all signs/amplitudes/cadences/cases, no best method/seed selection. Three MC cohorts are numerical integration replications, not independently trained models or different mechanisms. Source data, no training/validation/test optimization occurs; all settings fixed here.

Independent checks: NumPy fine recurrence and input schedules replay selected path IDs for all settings; independent NumPy score recomputation from all saved predictions; linear matrix-power input-operator and deterministic response calculation versus simulation. Save all source/script hashes, raw samples server-only, logs, statuses, compact tables and figures locally and remotely. Resource/time measured, no GPU dependency upgrades.

If exact passive maps plus endpoint inputs fail, but analytic input discretization or ordinary subcycling repairs them, reject **this evidence** as a need for a new world-model architecture. It is an interface/control-numerics confound. Even imperfect repairs do not prove an architecture gap because this audit fits no model and hides no state. A residual research direction would still require genuinely learned, partially observed nonlinear dynamics, same-information strong controls, independent physical mechanisms and held-out conditions. No final direction is accepted by the existence of this artifact.
