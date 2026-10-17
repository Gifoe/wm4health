# Pre-scoring ordinary stability control amendment

Added after observing validation-only long-rollout divergence (for example N87 seed2022 step2000 validation MSE about 1.59e18 while short training loss was about .0091), before any C60 neural test scoring. The immutable N87 test results from C59 were already known; this is an exploratory control completion, not an untouched preregistration. No C60 test values motivate or select this change.

All six original NSSM jobs keep their fixed seeds, budgets and selection rules. The first score waiter is explicitly superseded while verified WAITING with TEST_OPEN60_V1 absent; its process/state receipt is retained. Training owners are untouched. The replacement waiter must require both campaigns to terminate and all twelve attempts to be sealed before scoring.

Add a standard encoder-decoder GRU: history input [u,backward Delta u,y], GRU hidden30; future input [u,backward Delta u], separate GRU hidden30 initialized from the history state; scalar linear output after each future input. Exactly 6241 trainable parameters, versus native-component NSSM 6312 (about 1.1% fewer); latent dimension and update mechanism differ. This is existing seq2seq gated recurrence, not a new architecture or a pure stability-only ablation.

Use both systems, seeds 2021/2022/2023, fit-only normalization and the same train/validation/test partitions. Same sampler, observed history100, free-running future128, batch32, Adam lr .0005, 5000 updates, clip norm1, full validation every250 including0, best finite validation checkpoint and retained terminal. All outcomes remain visible, including any nonfinite fit. No new test-based recipe or hyperparameter choice.

Execute the fused native PyTorch GRU on CUDA while the original small scripted NSSM continues on CPU. Numerical wiring and CPU/CUDA gradients are qualified on synthetic data before real fitting; synthetic timings and optimizer steps are separate from real benchmark updates. Device choice is a resource scheduling decision, not a benchmark superiority claim. No packages or drivers change.

Only conventional bounded recurrent state and training/inference behavior are being checked. A GRU repair would reduce the need for a new long-memory/stability story here. Its failure would still leave horizon, initialization, coverage and finite-budget training explanations; it would not automatically establish a research gap.
