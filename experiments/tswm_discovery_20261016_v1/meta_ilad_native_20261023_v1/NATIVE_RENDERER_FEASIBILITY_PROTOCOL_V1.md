# Qualify the original forward CUDA kernels before any physical-model claim

The current Torch2.8 CUDA runtime contains NVRTC DLLs on the existing RTX5090 host. This permits checking whether the original gradsim forward kernels can execute with an in-memory NVRTC/driver binding, without installing OS components or a full compiler toolchain. It is an engineering option, not a new method.

Acquire only relevant source and licenses at gradsim commit abe8008e9559957f84a67b06208600cb66de3be2 directly on G; retain Git blob SHA and SHA256. Inspect all executed code. Preserve the native CUDA numerical kernels, launch arguments, mesh, camera, projection and original forward defaults. Any Python binding changes must be isolated, documented and independently checked. Do not replace rendered inputs with circles or another renderer and call the shipped checkpoint native-qualified.

First exercise one minimal NVRTC/driver launch against Torch's current context and stream. Then execute the original renderer's required forward kernels. No backward training is needed for data generation. Compare small deterministic fixtures against an independent scalar implementation, check camera/mesh transforms and batching invariance, then compare registered full-set tensor fingerprints and sums with source references. A source checksum mismatch is retained; aggregate-sum agreement alone cannot establish pixel equality. Original-paper bitwise parity and qualified functional execution must remain separate claims.

Do not launch model training during this qualification. If the runtime cannot execute the original kernels or input fidelity cannot be defended, retain the failure and choose a different native baseline. No source/driver/shared-runtime files are overwritten.
