# Import-isolation recovery and the dataset helper's declared batch requirement

The first source-editor harness terminated with exit1 at the dataset helper because AST isolation omitted its module-global alias `tree_map`. This is a harness error, not a missing symbol in the upstream module. Original outputs and terminal/log receipts remain untouched; recovery uses new execution/output/artifact directories and adds the original JAX tree-map alias.

Inspection of the same helper establishes an additional input requirement: it drops the last frame then reshapes to a hard-coded leading dimension10. For that helper only, register a B=10,T_input=65 fixture, yielding B=10,T=64 after the unchanged source transform. Other four helpers retain B=1/2,T_input=64. Both the native broadcast and explicitly non-native axis fixture remain. No helper is dropped, no edit magnitude or random seed is selected from observed output, and the source is unchanged.

The former generic B=1/2 fixture was not sufficient to exercise this batching convention. The recovery reports that transform as an interface fact, without concluding that the actual environment's trajectories were necessarily mixed. A valid physical-mechanism audit would separately preserve episode/parameter identity through any such batching.
