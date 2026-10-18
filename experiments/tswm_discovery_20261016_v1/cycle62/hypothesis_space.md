# Competing explanations

1. Finite-horizon optimization: longer rollouts change the use case; compare competent native multistep SSM training before attributing a deficit to missing model state.
2. Interface error: native batch rolling and an independent same-core loop must agree; pre-test training-history checks are retained.
3. Random approximate attention: report three fixed inference RNG streams separately from training-seed variation.
4. Ordinary inductive bias: ridge and seasonal controls may outperform neural arms without requiring a new mechanism.
5. Genuine missing capability: requires an independent mechanism/dataset and matched information/budget controls; this screen does not establish it.
