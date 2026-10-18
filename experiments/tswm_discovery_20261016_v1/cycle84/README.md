# C84: qualify the shift-adaptation research question

Read [the results](experimental_results.md), [prior-art boundary](literature_review.md), [negative evidence](negative_evidence.md) and [current research judgment](final_research_direction.md).

This cycle completed an original numerical qualification and primary-source audit. It did not reproduce a neural adaptation baseline or establish a new research direction. The established conjugacy construction prevents an unanchored component label from being treated as a universal ground truth.

For a fresh numerical run, use Python 3.10 with NumPy and SciPy, place a STORAGE_LOCATION_V1.json containing an artifact_root entry in the project parent of cycle84, and point it to a fresh external artifact directory. Run scripts/qualification84_v1.py, then scripts/verify84_v1.py. Both scripts derive the project root from their saved location. Existing evidence and artifact directories are protected against overwriting; use a fresh checkout/output location for reproduction. The source-contract audit caches are required only for building the publication snapshot, not for reproducing the numerical qualification.

The frozen protocol is QUALIFICATION_PROTOCOL84_V1.md. Public evidence uses logical server artifact locations; raw signal, checkpoint and prediction payloads are not included. Four control seeds are specified; no neural training seeds are represented by the witness counts.
