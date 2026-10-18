# Native supervision boundary

The two matched-mechanism arms share exactly the32 independent field-query locations and their labels. Native `_training_forward` additionally appends64 observation-derived field-query locations. Those appended locations and their synthetic field labels change when contexts change. Thus the complete native supervised query distribution is not fixed across the two arms. Both total query budgets and the underlying mechanism IDs remain identical.

This is an intervention on context generation in the actual native training pipeline, including its observation-derived supervision. A positive result would not distinguish an input-association shortcut from increased supervised state-space coverage. A separate fixed-query-supervision control would be needed before making that attribution. No such attribution is permitted from C87 alone.

The initial public checkpoint saw a much larger polynomial prior and longer/more trajectories. C87 adaptation uses two short context paths and bounded nonlinear systems. Consequently this experiment is neither a controlled explanation of the public checkpoint's pretraining failures nor a full replication of its original benchmark. It tests the proposed data mechanism in a native learner under a new controlled task.
