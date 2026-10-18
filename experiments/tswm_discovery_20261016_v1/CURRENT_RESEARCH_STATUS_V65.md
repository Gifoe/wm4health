# Current research status V65

Overall goal ACTIVE / UNACHIEVED. Accepted final research directions:0.

C65 executed18 native-CSDI/ordinary-MDN fits on6 independently seeded corpora across two known feedback systems, retaining240 generated laws and480 paired obs/do score views. All36 validation-budget checkpoints and all law metrics passed independent replay. See [cycle65](cycle65/README.md) and its complete tables; this is a transferred architecture mechanism audit, not original CSDI benchmark reproduction or a new framework.

Block conditioning, frozen-prefix g-computation, query-mask coverage, causal density models, exact iid references and exact linear denoiser/sampler controls separate semantic, training, sampling and finite-population effects. The native50-step sampler itself loses about12% variance even with an exact linear denoiser. Generic causalization, masking or reverse-variance fixes are established methods; no novelty is claimed from them.

Runtime disclosure: original training suffered0xc0000005 during checkpoint serialization after12 complete fits, leaving an invalid zero-byte checkpoint. A recovery import also failed; two isolated recovery attempts later failed in Tensor pickle serialization. Isolation excludes an injected old baseline venv path, retaining PyTorch2.8.0+cu128 with intended NumPy2.2.6. Four fits recovered with CPU PT snapshots and separate validation; a numeric NPZ/JSON backend with302-tensor exact roundtrip recovered the final two, resuming one at4000. All12 old fits passed common-runtime replay. Root cause remains unknown. The evaluator additionally hit0x80000003 after101 laws; the identical script resumed from verified receipts and all240 laws passed independent replay. EXECUTION_IDENTITY65_V3.json records actual final source identities. Successful formal updates144000; discarded original>=8000 and duplicated recovery>=4000, fixtures separately logged.

The standard causal MDN reaches do-distances0.078-to0.085 while coverage-prefix CSDI gives0.223-to0.543. This is negative evidence for a necessary new causal-generation architecture; it does not support a general impossibility claim.

The next source/weight audit targets an actual released pretrained framework, NormWear-2, with explicit distinction between observational completion and causal simulation. Staging or API fixtures do not constitute a performance result. Pythia's primary report still marks code/model to be released. Do not force an idea from existing causal-generation ingredients or software bugs; stronger independent-mechanism/original-task evidence is still missing.

All older C60-C64 artifacts and unrelated projects are preserved. Raw arrays/checkpoints remain on server G storage. GitHub receives compact code, reports and receipts; no result files are synchronized locally.
