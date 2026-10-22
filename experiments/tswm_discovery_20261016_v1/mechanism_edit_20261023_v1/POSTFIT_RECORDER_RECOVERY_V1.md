# Post-fit timestamp recorder recovery; no model restart

The actual factual child terminated with exit0 after2000 intended updates (2500 actual calls including the original incomplete500-call run). The post-fit queue then raised TypeError before any new verification child: its event helper supplied utc while also expanding a terminal receipt containing utc. This is our recorder bug, not a training, model or metric failure. Original queue FAILURE and trace remain unchanged.

Recovery renames an incoming receipt's utc to source_utc before attaching the fresh event timestamp. It reads the same completed training/checkpoints, reuses already completed exact metric verifications, and starts no training. Separate queue output paths preserve the failed recorder. All numerical thresholds, source hashes, checkpoint choice, clock diagnostic and counts remain unchanged.
