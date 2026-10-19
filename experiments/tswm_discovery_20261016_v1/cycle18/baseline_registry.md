# Baseline status: general time-series dynamical simulation

Native DynaMix source: DurstewitzLab/DynaMix-python39cc55514a4c5a0c50534c092fd9c3d40abdabc8. All22 selected src/license/config-description files preserved, GPL3; full tree contains no AGENTS. Official demonstration notebook and three data arrays are pinned separately with Git-blob and SHA256 verification. Model10,256 parameters,M30,N3,10AL-RNN experts,P2; learned state-attention noise remains active at evaluation.

Checkpoint: dynamix-3d-alrnn-v1.0.safetensors44,152B,SHA c950aef03f24e600bb159a05bf64aaea0fc89a6c66310baac624ac4b7e10ed5b. Config441B,SHA faf9e84a6a93b13d929e20ebdcf8963bab7a6dca1d0462e9513f98d74cadf261. HF mirror commit32f5adba587aab7b5eef10675f04eed2472dc12a verified against its Git/LFS metadata; direct official API timed out. Do not invent independent official-digest verification. Safetensors only, no pickle.

Executed: strict native interface preflight; nine native demonstration rollouts; independent metric/source/RNG verification. Not executed: full54-system paper evaluation, new training, intervention/control task, or probabilistic calibration of DynaMix. It is an autonomous simulator and is not required to expose actions or a Bayesian posterior.

Used existingE:/Anaconda/python.exe3.13.9,torch2.11.0+cu128,np2.3.5,CPU; meets declared>=3.11. No installs or unrelated-environment changes. Process-local single-thread profile. Main3.10 env and earlier sealed cycles unchanged.
