"""PU objectives and deterministic simplex optimization for FedNearPU v2."""
import numpy as np
import torch
import torch.nn.functional as F


def nnpu_marginal_loss(logits_p, logits_d, pi_d):
    from utils.nnpu_risk import corrected_nnpu
    return corrected_nnpu(logits_p, logits_d, pi_d)


def cached_distillation_loss(logits_u, q, r):
    """Confidence-weighted soft BCE used only as an auxiliary regularizer."""
    lu = logits_u.reshape(-1)
    q = q.reshape(-1).detach().to(device=lu.device, dtype=lu.dtype)
    r = r.reshape(-1).detach().to(device=lu.device, dtype=lu.dtype)
    if not lu.numel():
        return lu.new_zeros(())
    return (r * F.binary_cross_entropy_with_logits(lu, q, reduction='none')).mean()


def cached_soft_loss(logits_p, logits_u, q, r, lambda_u=1.0):
    """Positive BCE plus per-example weighted soft-label BCE on U."""
    lp = logits_p.reshape(-1)
    lu = logits_u.reshape(-1)
    q = q.reshape(-1).detach().to(device=lu.device, dtype=lu.dtype)
    r = r.reshape(-1).detach().to(device=lu.device, dtype=lu.dtype)
    if lp.numel():
        loss_p = F.softplus(-lp).mean()
    else:
        loss_p = lu.new_zeros(())
    if lu.numel():
        loss_u = (r * F.binary_cross_entropy_with_logits(lu, q, reduction='none')).mean()
    else:
        loss_u = lp.new_zeros(())
    return loss_p + float(lambda_u) * loss_u, {
        'loss_p': float(loss_p.detach()), 'loss_u': float(loss_u.detach())
    }


def project_simplex(v):
    """Euclidean projection onto {x >= 0, sum(x)=1}."""
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    if v.size == 0:
        raise ValueError('Cannot project an empty vector onto the simplex')
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    idx = np.arange(1, v.size + 1, dtype=np.float64)
    valid = u - cssv / idx > 0
    rho = np.nonzero(valid)[0][-1]
    theta = cssv[rho] / float(rho + 1)
    return np.maximum(v - theta, 0.0)


def solve_pu_brier_weights(pred_u, pred_p, alpha_u, tau=0.1,
                           min_p=5, min_u=32, max_iter=2000, tol=1e-6):
    """Solve the regularized PU Brier objective over teacher probabilities.

    Prediction matrices have one row per sample and one column per candidate.
    Returns (weights, diagnostics); insufficient or invalid scores return the
    uniform fallback with a machine-readable reason.
    """
    u = np.asarray(pred_u, dtype=np.float64)
    p = np.asarray(pred_p, dtype=np.float64)
    if u.ndim != 2 or p.ndim != 2 or u.shape[1] != p.shape[1]:
        raise ValueError('pred_p and pred_u must be 2D matrices with matching teacher columns')
    m_teachers = u.shape[1]
    if m_teachers == 0:
        raise ValueError('At least one candidate teacher is required')
    uniform = np.full(m_teachers, 1.0 / m_teachers, dtype=np.float64)
    reason = None
    if p.shape[0] < int(min_p):
        reason = 'insufficient_p_score'
    elif u.shape[0] < int(min_u):
        reason = 'insufficient_u_score'
    elif alpha_u is None or not np.isfinite(alpha_u) or not 0 <= float(alpha_u) <= 1:
        reason = 'invalid_prior'
    elif not np.isfinite(u).all() or not np.isfinite(p).all():
        reason = 'nonfinite_predictions'
    elif tau <= 0 or not np.isfinite(tau):
        raise ValueError('tau must be positive and finite')
    if reason:
        return uniform, {'fallback': True, 'reason': reason, 'iterations': 0,
                         'gap': None, 'objective': None}

    alpha_u = float(alpha_u)
    C = (u.T @ u) / float(u.shape[0])
    m = p.mean(axis=0)
    # Symmetrize tiny floating-point roundoff before eigensolving.
    C = 0.5 * (C + C.T)
    eig_max = max(0.0, float(np.linalg.eigvalsh(C)[-1]))
    step = 1.0 / (2.0 * (eig_max + float(tau)))

    def objective(a):
        return float(a @ C @ a - 2.0 * alpha_u * (m @ a) + alpha_u
                     + tau * np.sum((a - uniform) ** 2))

    a = uniform.copy()
    gap = np.inf
    for iteration in range(1, int(max_iter) + 1):
        grad = 2.0 * (C @ a - alpha_u * m + tau * (a - uniform))
        gap = float(grad @ a - np.min(grad))
        if not np.isfinite(gap) or not np.isfinite(grad).all():
            return uniform, {'fallback': True, 'reason': 'nonfinite_solver',
                             'iterations': iteration, 'gap': None, 'objective': None}
        if gap <= tol:
            return a, {'fallback': False, 'reason': None, 'iterations': iteration,
                       'gap': gap, 'objective': objective(a)}
        a = project_simplex(a - step * grad)
    return uniform, {'fallback': True, 'reason': 'solver_max_iter',
                     'iterations': int(max_iter), 'gap': float(gap),
                     'objective': objective(uniform)}
