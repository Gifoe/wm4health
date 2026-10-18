# Execution qualification repair

No formal training campaign was launched before qualification passed. Initial fixture v1 reused a side-embedding autograd graph across updates and failed after one successful update; regenerate side embeddings at each iteration (the formal train script already did so). Fixture v2 passed 100 finite updates but did not meet the fixed 2e-4 CPU/GPU comparison guard. Neither attempt produced a PASS receipt or performance result.

Fixture v3 disables CUDA matmul and cuDNN TF32 before model construction and keeps highest float32 matmul precision. The same 100-update test then passed the unchanged guard, with max output difference2.384185791015625e-7 and side-embedding difference5.960464477539063e-8. This is a numerical configuration qualification, not a claim that all previous native crashes or precision issues are fixed. Actual formal campaign script is train65_v2.py, which applies the qualified settings. Native upstream CSDI bytes, model algorithms and objectives are unchanged; no dependency stubs or replacement architecture are used.

Qualification artifacts are untrained diagnostic fixtures and are not counted among the18 formal models.
