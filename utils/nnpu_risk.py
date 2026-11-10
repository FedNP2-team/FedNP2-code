"""Shared nnPU marginal risk, including negative-risk corrective gradients.
D is the full observed marginal P union U; pi is its positive-class prior.
The reported loss and the objective used for backpropagation are both logged.
"""
import torch
import torch.nn.functional as F

REPAIR_VERSION = "nnpu_corrective_v1_20261025"

def corrected_nnpu(logits_p, logits_d, prior, beta=0.0, gamma=1.0):
    lp, ld = logits_p.reshape(-1), logits_d.reshape(-1)
    pi = float(prior)
    if not lp.numel() or not ld.numel():
        raise ValueError("nnPU requires non-empty observed P and marginal D batches")
    if not 0.0 <= pi <= 1.0:
        raise ValueError(f"Invalid marginal class prior: {pi}")
    if beta < 0 or gamma <= 0:
        raise ValueError("nnPU correction requires beta>=0 and gamma>0")
    pp = F.softplus(-lp).mean()
    pn = F.softplus(lp).mean()
    dn = F.softplus(ld).mean()
    positive = pi * pp
    negative = dn - pi * pn
    corrected = bool((negative.detach() < -float(beta)).item())
    if corrected:
        reported = positive - float(beta)
        optimize = -float(gamma) * negative
        loss = reported.detach() + (optimize - optimize.detach())
    else:
        reported = optimize = positive + negative
        loss = optimize
    parts = {
        "r_p_pos": float(pp.detach()), "r_p_neg": float(pn.detach()),
        "r_d_neg": float(dn.detach()), "r_n_raw": float(negative.detach()),
        "pu_r_positive": float(positive.detach()),
        "pu_r_negative_raw": float(negative.detach()),
        "pu_r_negative": float(negative.detach().clamp_min(0)),
        "pu_correction_active": int(corrected),
        "pu_reported_objective": float(reported.detach()),
        "pu_optimization_objective": float(optimize.detach()),
    }
    return loss, parts
