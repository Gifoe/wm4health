# Validation-launch prerequisite correction

The first new validation launcher referred to the old raw-control registry path while expecting the new decoded-control terminal status. Its prerequisite guard rejected execution before creating a scientific child process. All six decoded training runs had already completed with exit0; none was repeated.

The V2 launcher changes only that registry path to `outputs/decoded_control_v1/RESULTS_V1.json`. The validation script, frozen weights, data, metrics, selected epochs, Validation-only branch criterion and three-paper scope are unchanged. V1 remains retained. This is an execution mistake, not a model failure or research-gap observation.
