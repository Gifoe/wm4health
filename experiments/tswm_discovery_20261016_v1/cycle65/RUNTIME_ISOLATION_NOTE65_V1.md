> Superseded source-origin attribution: read RUNTIME_ISOLATION_NOTE65_V2.md. The original text is retained below as historical evidence.

# Runtime provenance and isolated execution

Two native failures are verified: original training failed during step8000 checkpoint creation (zero-byte file), and the first recovery coordinator failed during torch import before any model training. The second failure's faulthandler stack reaches torch._ops._get_packet via torch._meta_registrations. Windows reported unknown fault module for the second failure. Root cause is not established.

A concrete environment difference was found: normal startup imported NumPy1.26.4 from user-site, while Python-I in the intended conda environment imports NumPy2.2.6. Isolated execution disables user-site/PYTHONPATH and supplies the conda runtime's DLL directories explicitly. PyTorch remains2.8.0+cu128; no environment, dependency or driver is installed or changed. The isolated100-update fixture passes the same CPU/GPU guard (2.384185791015625e-7) and hidden-value invariance guard(0). This does not prove that all long native failures are fixed.

The full-attention CSDI configuration has is_linear=False. Its optional linear_attention_transformer dependency was present only outside the isolated environment. The isolated entry loader omits precisely its unused import by AST; original pinned bytes and every function/class body remain unchanged. It supplies no fake dependency and never invokes the unselected linear-attention branch.

Recover campaign v2 uses isolated children, unchanged architecture/loss/data/budget/validation selection, CPU-only atomic snapshots, and complete RNG state. All12 successful original fits remain; final independent replay must revalidate their metrics under the common isolated runtime. Existing data arrays are verified byte-for-byte against seeded regeneration before remaining work. Any numerical/software difference and failed compute stays explicit; no performance claim can use the old in-progress state as completion evidence.
