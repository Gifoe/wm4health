# C46 evaluation counts

There are **1280 unique EVAL trajectories**:512 from the four held-out temporal combinations (128 each), and768 from the twelve seen combinations (64 each).

Each trajectory is scored for twelve predictors: eight independently fitted neural model instances (four arms × two initializations), one aligned PCA/LGSSM, one empirical whole-path distribution, zero, and persistence. Thus the verifier checks **15360 model–origin score entries**, stored in192 model–cell raw groups. The repeated appearances of one trajectory across predictors are paired comparisons, not additional independent trajectories. Each stochastic forecast uses64 joint future draws; draws are not training seeds or independent evaluation origins.

No population-level stability claim is made from two initializations or four fixed held-out system/view cells. The point controls have no meaningful probabilistic interval coverage. The complete counts do not imply completion of the autonomous research goal.
