# Baseline registry, C87

- Native FIM-ODE, public checkpoint HF FIM4Science/fim-ode revision00875c253d025dd33fd70ae751089b1be3a7c174;51.9MB,12,968,068 parameters; previously qualified and hashed C80. Unchanged pinned source cee2bb53. Frozen inference and9 complete-parameter controlled adaptations:3 data arms x3 seeds.
- Ordinary weak-form polynomial identification degrees1/2/3 x5 ridge values. Uses only the same noisy context at inference. VAL-only selection is frozen before evaluation. Not given family/parameter labels. All15 choices retained.
- Persistence, same query initial state, no future observations.
- Exact Duffing/Stuart-Landau fields and DOP853 references are simulator validation/oracles, not learned competitors.
- C71 original-budget modular meta-learning continues separately. It is not a completed row in C87 and is not a controlled-training comparison with FIM.

Task class: autonomous dynamics inference and free simulation from a new initial state. No action-conditioning or causal-treatment claim. C87 is a custom controlled experiment using actual native code/weights, not an original-paper leaderboard reproduction.
