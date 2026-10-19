# Cycle4 baseline registry

| Source or control | Revision / definition | Execution status |
|---|---|---|
| mkarmann/motion-jepa | b3c2fb374b7c69f169de15774f371290a8736f56 |65pinned code/config files; native objectives with declared vector adapters:6pilot+5attribution fits |
| YongchaoHuang/VJEPA |88a90d2c0af970edd46d44d53d4864d10aa6d117 |11small text files; native CNN/loss/CEM component checks at3seeds; no native training |
| Linear CCA/VAMP-style | Ordinary whitening and lagged cross-covariance SVD | Training-only initialization control; not a new method or third-party package reproduction |
| Heteroscedastic matched controls | Same4549parameter observation-target model; variance input detached/attached |9fits incl3full-observation controls; new diagnostic, not VJEPA paper replication |
| Training predictive cross-moments | OLS mean and Cov(x,a*residual^2) directions |3fits, no true state/channel directions used for fitting; ordinary-method repair control |

Prior LatentTSF/DAG/WorldTS original-paper reproduction evidence remains in the
earlier registries. RP-GSSM execution belongs to cycle2; EpiCF/LeNEPA checks to
cycle3. No cross-protocol pooled leaderboard.10primary PDFs are hash mirrored.
Runtime verified:Python3.10.20,Torch2.8.0+cu128,NumPy1.26.4,RTX5090.
Motion project declares a newer full-stack runtime; standalone objective imports
are not confirmation that its full paper training stack is reproduced here.
