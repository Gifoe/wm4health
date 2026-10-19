# Continuation after cycle 3

The overall autonomous discovery goal is ACTIVE and NOT ACHIEVED. No accepted
research direction. This is a reproducible evidence checkpoint, not completion.

Read outputs/task_status.json, experimental_results.md, negative_evidence.md and
final_research_direction.md before running more experiments. Preserve cycle 1/2.

Completed here: public EpiCF v1.2 files downloaded on the server with published
MD5 and SHA256 verification; native preprocessing for single and multi settings;
13 single-policy baseline configurations (1 KDE, 3 Transformer, 9 matched-capacity
input ablations); 3 native LeNEPA component checks. All logs/weights remain server.
The local mirror intentionally omits the heavy weights, prediction arrays and raw
data; they are recorded by the server manifest. No content has been published.

Findings: EpiCF test inputs include the target counterfactual S/E/R/M trajectory
for the whole sequence. Policy-blind retrained models improve its released scores.
The 158 blocks contain only 153 unique counties, with 5 exact duplicates. This is
a specific task/protocol diagnostic, not a proven novel model gap. Do not use this
as a history-only forecasting leaderboard. Simple susceptible-difference exact
reconstruction failed; keep the negative result.

LeNEPA's official source is pinned at 6574eb9e123305334e09e180de67682844f2e61f.
The abstract position-table null is not admissible as its default input-layer
target (no absolute PE; layer 0 and layer 8 both regularized). Native RoPE-only
components retain constant time input within numerical precision. Do not claim
the whole model fails from a tensor-level counterexample. No full LeNEPA run.

New close work to account for: Semigroup-JEPA2609.10464, One-Step Next-Latent
Prediction2609.36227, MotionJEPA2609.23881, TC-LeWM2607.26924, author ChronoJEPA
repository, controlled-world-model identifiability2607.22430 and deployable action
geometry2610.16644. Publication/primary-source depth differs; see literature notes.

Do not redo RPM calibration tuning, add action memory as a claimed invention,
rename a conditional-mean Jensen gap, or invent a method merely to finish. Next
steps should distinguish a new, concrete failure from these existing mechanisms.

Reproduce source staging via scripts/stage_epicf_sources.py, preprocessing via
audit_epicf_protocol.py, training via run_epicf_baselines.py, component checks via
audit_lenepa_position_null.py, summaries via summarize_epicf_audit.py. Run with
the original discovery .venv, Python -B; standalone component imports avoid
executing the full external training stack. Local source downloads were securely
mirrored when the server's ordinary GitHub route refused connections; data itself
was downloaded directly on server with normal HTTPS certificate verification.
