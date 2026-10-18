# Model and evidence registry

| Model/control | Executed | Status |
|---|---|---|
| Native source-pinned CSDI | One seed,20000 updates,413569 parameters | Duffing CAL support passes; reservoir fails; later seeds and EVAL not opened |
| Empty-component-pruned unsupervised mixture | Same C89 fitted coefficients, TRAIN-only support repair | Both CAL and independent confirmation pass |
| Group-supervised polynomial library | Two closed-form TRAIN fits | Both CAL identification accuracies100%; independent confirmation passes |
| Flat MLP interpreter | One seed,10000 updates | CAL selection by CE gives accuracy0.714844, macro0.533467; disqualified |
| Native semantic coupling comparison | Code/protocol authored | NOT executed because the generator did not qualify across both families |

Native source is unchanged `ermongroup/CSDI@7f24a436f08d98853a6b43d4f7f04e5a65ecdf27`. It is a controlled cross-task transfer, not original-paper table reproduction. TRAIN8192/CAL256 arrays and original normalization/masks remain unchanged. The full-budget fit takes1068.594s; the selected step is20000. No new environment installation occurred.

The unsupervised safeguard removes Duffing components[0,4,8], total mass2.99999101619e-06, solely because expected TRAIN membership<1. All retained coefficients are byte-exact to the failed original checkpoint; no trajectory is removed or clipped. Reservoir retains all components. The original failure remains in C89.

The library uses TRAIN group identities, known component count, a degree5 fully observed Markov basis and family-specific parameters. The unsupervised mixture also has strong Markov/polynomial bias. These are positive controls, not a matched-information/capacity/compute leaderboard against CSDI. Deployment identification receives observed/generated states and actions, not physical parameter values or hidden class IDs.
