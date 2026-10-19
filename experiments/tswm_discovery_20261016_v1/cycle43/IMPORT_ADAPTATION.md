# Isolated import adaptation

The original import attempt failed before model construction because the MMDiT package initializer re-exports an unused SD3 text encoder, requiring Transformers. The SPF backbone and numerical sampling path do not use that text encoder. The original source archive remains immutable.

`prepare_source43.py` copies the29 verified files into a runtime directory and removes only the unused text-encoder re-export from the package initializer. Backbone, blocks, embeddings, normalization, schedulers, collator and sampling/training function bodies remain byte-identical. The adapter manifest records original and runtime SHA256. This is disclosed source adaptation, not a complete original-environment reproduction; no numerical behavior or pretrained weights are substituted.

Five original-PyPI wheels are installed only in cycle43/vendor: diffusers0.31.0, accelerate1.1.1, huggingface-hub0.32.3, tensorboardX2.6.4 and regex2024.11.6. Process-local import precedence overrides the global newer Hub version. Global packages and unrelated jobs are unchanged. The initial pip network failure was resolved using official PyPI JSON/download URLs with normal TLS and proxy inheritance disabled, without TLS bypass or a third-party mirror.
