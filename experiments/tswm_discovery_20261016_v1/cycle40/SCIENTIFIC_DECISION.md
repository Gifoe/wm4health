# C40 scientific decision after actual results

The result does **not** support “existing time-series models cannot effectively incorporate new observations.” Both ordinary families improve their mean fixed-tail energy score after 32 actual prefix observations on both carriers:

| Carrier | GRU-mixture r0 -> r32 | Native FlowNP32 r0 -> r32 |
|---|---:|---:|
| ETTh1 | .224754 -> .204459 | .375848 -> .346325 |
| Jena | .331656 -> .274464 | .304651 -> .269350 |

These are two-seed exploratory VAL means, not a universal theorem or independent confirmation. Jena supplies only five temporal blocks; both datasets are temperature targets. The effect of eight observations is smaller/more uncertain in several cells. All cells, scores, seeds and paths remain recorded.

The quality/cost comparison also has strong ordinary counterexamples to a general bottleneck claim. On ETTh1, the recurrent non-Gaussian joint baseline has mean energy .216241 at approximately .0604 s/query, versus native FlowNP32 .365083 at .1381 s. On Jena, FlowNP8 reaches .293668 at .0371 s, while the GRU-mixture has .298498 at .0621 s. FlowNP32 improves to .287514 while costing .1484 s; steps1/2 are faster with lower quality. These comparisons describe this API/configuration and single-origin synchronized 32-draw calls; they do not characterize all state-space or generative models. No method dominates every carrier and cost point.

Accordingly, close the **broad failure claim** behind a generic “add observation correction” proposal. There is no justification here for inventing an updater, tuning one site until it wins, or renaming an ordinary filter. Do not treat a selected noisy curve or a quality/cost tradeoff as evidence of a novel framework mechanism. Two fits select the cap and six have a final validation improvement relative to the previous checkpoint; capacity/optimization remain unresolved for stronger comparative claims.

The user's framework scope remains in force. The more precise next research question is about **what predictive-state learning contributes beyond ordinary simulation/inference machinery**, rather than whether an update interface can exist. A useful future mechanism experiment must separate (i) the information contained in the learned state, (ii) the transition approximation and (iii) posterior/inference error, using known-state truth and exact or high-accuracy ordinary controls. Only a demonstrated interaction between state learning and useful dynamics would justify a new architecture. This is an evidence-development priority, not an accepted idea.

Recent LIFTS now supplies an executable-looking close reference with joint latent generation/inference, informative observation masks and synthetic encoder calibration already considered. Its author source is pinned and cached, but has **not** been run. Its native semantics and assumptions should be qualified before any claim that it lacks the needed capability. SFM similarly remains a known ordinary control if a specific cheap-update residual is later identified. Do not automatically exhaust more data/seeds to rescue the generic update claim.

Research outcome: PROGRESS. Accepted novel final directions: 0. Overall objective: ACTIVE_UNACHIEVED. Existing models provide substantive negative evidence against the broad current motivation. The first screen is complete; no new method, paper-ready novelty or TEST result is claimed.
