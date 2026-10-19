# Frozen public generative baseline qualification

This is Phase-A executable qualification, not an accepted research idea or a scientific failure demonstration. The three C34 perspectives identify strong direct prior art for ordinary latent importance sampling. No new proposal is trained in this qualification. A later method experiment requires a separately frozen discriminator against same-parameter conditional mixtures and competent classical/RQMC controls, and must justify its contribution to reusable world modeling.

## Source, data, runtime and storage

- Sundial author model `thuml/sundial-base-128m`, revision3212e42564493f520593e5414af4367fc4b49226. Seven small public source/config/card files have been downloaded and Git-blob/SHA256 verified locally. Source inspected before import; no launcher, credentials or unrelated project is executed.
- Expected safetensors bytes513341448, SHA256414435b508391f92afadd2aaeec418c806776aeccbce12e638d73a139ca5ca78. Download directly to the new server C34 assets folder, HF mirror first, normal TLS. Existing correct file is reused; a mismatched existing file stops the operation. No local weight copy, no deletion or overwrite of old artifacts.
- Two existing public FEV parquet sources, dataset revisionf71c0fff4cf81283a2c43e7f3a73aa4f9826aef8: `solar_with_weather/15T/train-00000-of-00001.parquet` and `jena_weather/10T/train-00000-of-00001.parquet`. Verify exact API identities, sizes and Git-LFS SHA before use. Preselect `target` and `target_0`, respectively. No future covariates or labels enter this qualification.
- Existing remote PyTorch2.8+cu128 environment has no transformers/tokenizers and HFhub2.2.0. Use a task-local vendor folder, preserving all old packages. Pin transformers4.40.1, tokenizers0.19.1, huggingface-hub0.23.5 and regex2024.5.15 from official PyPI wheels; verify published SHA256 and safe archive paths. Record actual import versions/paths and all other dependencies. No global pip operation.

## Native contract and parity

Native generate accepts univariate[B,L], outer normalization by history mean and population std+1e-5; direct-forward RevIN uses another rule and must not be substituted. Config emits one full720-vector from720-dimensional Gaussian noise, then crops H<=720. Preserve all50 sampling steps and `x += (net(x,t,condition)-original_noise)*dt`. No token truncation of the source, no reduced steps, fp16/TF32 substitution or covariate input.

First run the official source-native `generate` in eval/float32 on a deterministic numerical fixture and the first eligible real histories of both preselected public fields (history672, H96). A real history is the earliest contiguous finite length672 sequence under the recorded source ordering; do not select it by model output. It is not a held-out forecast score or a new dataset-tuned method.

Expose full Gaussian noise and cache the exact history conditioning. Compare this route to the native sampler and native API using the identical CPU Gaussian random bank and its exact sample/batch ordering. Require finite outputs and relative max error<2e-5, verify batch/prefix replication within float32 tolerance, model state unchanged, and no additional input. Save inputs, noises, source conditioning, outputs and manifests. A fresh independent script replays these checks and independently recomputes source/file identities before claiming qualification.

Also verify a smooth output functional's noise gradient against central directional finite differences, using step0.01 and relative tolerance0.03, solely to qualify any future gradient-based ordinary control. This is not a dynamics correctness result.

H>720 native continuation feeds the ensemble median into shared history; do not treat those samples as independent ancestral paths. This pilot restricts H96. A long-path inference modification would be labeled as such and is not a claimed research gap or proposed method.

## Decision

Pass means the released generator is a reproducible same-law numerical integration anchor on the server. It does not establish calibrated real-event risk, a new world-model mechanism, model superiority or final research-direction acceptance. Any source/runtime failure is preserved and fixed separately. No formal candidate TEST evaluation or long training begins here.
