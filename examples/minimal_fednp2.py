"""Demonstrate seed-0 probe rotation and PU-risk-weighted supervision."""
import numpy as np
import torch
from models.fednearpu_v2 import RotatingProbeManager
from models.utils.pu_objectives import cached_distillation_loss, solve_pu_brier_weights
from utils.nnpu_risk import corrected_nnpu

np.random.seed(0)
torch.manual_seed(0)
probes = RotatingProbeManager(np.arange(12), 4, 4, 2, seed=0)
print("Active probes:", probes.active_ids.tolist())
print("After rotation:", probes.rotate().tolist())
positive_predictions = np.random.uniform(0.5, 1.0, (8, 3))
unlabeled_predictions = np.random.uniform(0.0, 1.0, (40, 3))
weights, diagnostics = solve_pu_brier_weights(unlabeled_predictions, positive_predictions, 0.5)
soft_targets = torch.tensor(unlabeled_predictions @ weights, dtype=torch.float32)
confidence = (2 * soft_targets - 1).abs()
logits = torch.randn(40, requires_grad=True)
risk, _ = corrected_nnpu(torch.randn(8), logits, 0.5)
loss = risk + 0.2 * cached_distillation_loss(logits, soft_targets, confidence)
loss.backward()
print("Teacher weights:", weights.round(4).tolist())
print("Solver fallback:", diagnostics["fallback"])
print("Loss:", round(float(loss.detach()), 4))
