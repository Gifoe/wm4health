# Pinned LIFTS native execution qualification

This is source/API qualification, not headline reproduction or evidence that a new mechanism is needed. Reuse author source e42430980e51bebf2e1b54d1beddeda7a31dee51 read-only from C40. Fetch only missing import dependencies at the same pin into C41 supplemental source, verifying Git blob and SHA256. No global installs.

Run the author's case1_minimal.py settings: p20,d5,T48,MCAR.5,Ntrain80/Ncheck20,seed42,total_loops5,batch32,n_samples10,lambda_selfmask5,lambda_enc1,enc_warmup2. This example intentionally differs from the paper's main lambda_enc0; label that difference. The Ncheck partition is an interface fixture, not held-out method selection or a real TEST benchmark.

Save native config, source receipts, exact data and latent trajectories/parameters, fitted checkpoint, scaler, sample arrays, runtime/memory and stdout/stderr. Check shapes/finite values, observed-value hard replacement in imputation, replay under fixed RNG, and invariance to corruption where observation mask is zero. Characterize stochasticity and causal access rather than assuming comments are correct. A source error is a baseline qualification issue, not a research gap.

Construct a separate independently implemented exact Kalman posterior/control on the same **known** linear-Gaussian DGP, with correlated Q/R and exact finite burn-in prior. It is a privileged oracle to diagnose information and inference; it is not a fair unknown-parameter competitor. Verify recursive filtering against a block Gaussian conditioning implementation on short fixtures, and distinguish missing-at-random from informative mask settings (oracle only valid as written for MCAR).

No neural conclusion from five loops or tiny data. Do not invent an updater or loss based on this qualification. Before any new comparative training, use the source/controlled evidence and independent research notes to freeze a hypothesis that has not already been closed in C15/C25/C32–40.
