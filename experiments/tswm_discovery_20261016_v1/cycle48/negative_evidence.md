# C48 negative evidence and attribution limits

Deprioritize the current explicit-delay route as a proposed new framework. It provides no stable advantage over the competent history-aware controls. On Jena, GRU is better than learned delay in both seeds for open-loop and updated forecasts. On ETT, learned versus GRU depends on seed; the weak scalar NODE has the lowest mean open-loop MSE among these arms. This does not rule out useful memory generally or establish NODE as a universal world model.

Real new observations improve the same final24-hour target with ordinary controls: GRU tail MSE ETT0.0832203 ->0.0431851 and Jena0.1992946 ->0.1104120; ridge ETT0.0946744 ->0.0452926 and Jena0.2392916 ->0.1188755. Means are over two GRU seeds and one deterministic ridge fit. These are point-forecast updates, not Bayesian-law consistency proofs.

The stronger claim that existing models cannot incorporate fresh evidence is contradicted in these tasks. Learned delays are already an existing mechanism; their addition alone cannot supply novelty. No hard caps/final-step selections or numerical flags were observed, but optimization, capacity and two-process coverage still limit any universal conclusion. A scalar NODE doing well does not prove history is unnecessary in other systems.

Prior C47 fixed-feature Laplace/hybrid and C46 simple neural factorization remain deprioritized. New C49 is a different conditional learning-objective feasibility screen with fresh trajectories; it is not a retune of these EVALs.
