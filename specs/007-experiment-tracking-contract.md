# Spec 007: Experiment Tracking Contract

## Goal

Track substantive experiments consistently without turning tracking failures into training failures.

## Tracking Backend

The first tracking backend is Weights & Biases.

References:

- W&B run overview: <https://docs.wandb.ai/guides/runs>
- W&B Run API: <https://docs.wandb.ai/ref/python/experiments/run/>
- W&B logging guide: <https://docs.wandb.ai/guides/track/log>

## Auto-Enable Policy

Tracking should auto-enable when:

- `wandb` is installed
- tracking is not explicitly disabled
- a usable W&B credential is present in the environment

If tracking initialization fails, experiments must continue with a no-op tracker.

## Entity And Project Policy

- default entity: `achbogga-track`
- project names must stay stable across hyperparameter and performance tuning within the same
  architecture family
- materially different architecture families must use different project names

The project naming key must distinguish:

- dataset family
- view/modality family
- image-backbone family
- teacher-seed architectural mode

It must not distinguish:

- learning rate
- batch size
- gradient accumulation
- small query-budget or threshold tuning

## Required Coverage

The following experiment surfaces must emit tracking runs when tracking is enabled:

- `fit_nuscenes`
- `fit_openlane`
- `run_nuscenes_overfit_gate`
- every recipe run inside `run_bounded_research_loop`

## Required Logged Information

Each run must log:

- dataset and split metadata
- model/config payload
- training hyperparameters
- epoch-level train and validation metrics
- final checkpoint path
- final validation metrics

Research-loop recipe runs must additionally log:

- official eval metrics
- benchmark metrics
- source-mix diagnostics
- recipe decision status

Overfit-gate runs must additionally log:

- gate verdict
- train-total ratio
- official gate metrics

## Failure Policy

- tracking failures must not abort the experiment
- failed initialization should be reported once in stdout and then become a no-op
