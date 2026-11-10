# FedNP²

Core implementation for federated positive–unlabeled learning using neighborhood-aware personalized supervision.

## Contents

The release includes rotating shared probes, neighbor discovery, PU-risk-weighted teacher combination, cached supervision, nnPU optimization, FedAvg, the PU data protocol, and the ResNet model. The method code is based on the seed 0 `nnpu_combined` server implementation. 

`configs/seed0.json` records the shared settings of the seed 0 runs. It is a configuration reference. Dataset-specific probe sizes and partitions are supplied by the caller.

## Minimal example

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
PYTHONPATH=. python examples/minimal_fednp2.py
```

The example uses synthetic data to demonstrate probe rotation, teacher weighting, and a local loss. Integrating the core into a training loop requires datasets, client partitioning, loaders, and evaluation.

## Scope

This review-stage snapshot includes the method core. It does not include dtrained checkpoints, experiment logs, or scripts for running the complete experiment matrix. 
