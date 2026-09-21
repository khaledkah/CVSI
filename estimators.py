from __future__ import annotations

from typing import Callable

import torch

from dem.models.components.cvsi import (
    clip_scores,
    get_a2_b2,
    tilde_ct_opt_fn,
    tilde_ct_opt_weighted_fn,
)


def _grad_log_p(neg_energy_function):
    """d/dx0 log p(x0) via torch.func.grad (same construction as cvsi.py)."""
    return torch.func.grad(lambda x0: neg_energy_function(x0).sum())


def score_from_samples(
    t: torch.Tensor,
    x: torch.Tensor,
    x0_samples: torch.Tensor,
    w: torch.Tensor,
    h_t: torch.Tensor,
    neg_energy_function,
    noise_sch,
    estimator: str = "cvsi",
    weighted_stats: bool = False,
    ct_per_molecule: bool = True,
    clipper_tsi=None,
    return_ct: bool = False,
):
    """TSI / DSI / CVSI score from given posterior samples ``x0_samples`` and weights ``w``.

    Shapes: ``x0_samples`` (K,B,D), ``w`` (K,B,1), ``h_t`` (K,B,1), ``x`` (B,D). Mirrors the
    combination steps of ``cvsi.idem_score_cvsi_fn``, with c~*(t) computed by
    ``tilde_ct_opt_fn`` / ``tilde_ct_opt_weighted_fn`` from the unclipped TSI branch.
    ``clipper_tsi`` bounds only the TSI branch mixed into the returned score; DSI is never
    clipped. With ``ct_per_molecule=True`` on a molecular energy, c~*(t) is computed per
    particle and then averaged.
    """
    assert estimator in ("tsi", "dsi", "cvsi")
    if t.ndim == 0:
        t = t.unsqueeze(0).repeat(len(x))

    # a(t), b^2(t) of the kernel N(a x0, b^2 I): DSI = (a x0 - xt)/b^2, TSI = grad/a.
    a2_t, b2_t = get_a2_b2(t, noise_sch)                 # (B,)
    a_t = a2_t.sqrt().reshape(1, -1, 1)                  # (1,B,1)
    b2_t = b2_t.reshape(1, -1, 1)                        # (1,B,1)

    dsi_broad = (a_t * x0_samples.detach() - x.unsqueeze(0)) / b2_t
    if estimator == "dsi":
        return (w * dsi_broad).sum(dim=0)

    # Call under no_grad: torch.func.grad otherwise retains a graph through the
    # energy's own parameters. The gradient w.r.t. x0_samples is correct either way.
    with torch.no_grad():
        grad_log_p = _grad_log_p(neg_energy_function)(x0_samples)
    tsi_raw = torch.nan_to_num(grad_log_p / a_t, nan=0.0, posinf=0.0, neginf=0.0)
    tsi_broad = tsi_raw
    if clipper_tsi is not None:
        tsi_broad = clip_scores(tsi_raw, clipper_tsi, neg_energy_function)

    if estimator == "tsi":
        return (w * tsi_broad).sum(dim=0)

    if ct_per_molecule and getattr(neg_energy_function, "is_molecule", False):
        n_p, n_s = neg_energy_function.n_particles, neg_energy_function.n_spatial_dim
        K, B = tsi_raw.shape[0], tsi_raw.shape[1]
        tsi_r = tsi_raw.reshape(K, B, n_p, n_s).permute(1, 0, 2, 3)   # (B,K,n_p,n_s)
        dsi_r = dsi_broad.reshape(K, B, n_p, n_s).permute(1, 0, 2, 3)
        t_r = t.unsqueeze(-1).unsqueeze(-1)                            # (B,1,1)
        if weighted_stats:
            w_r = w.reshape(K, B, 1, 1).permute(1, 0, 2, 3)           # (B,K,1,1)
            _, tilde_ct = tilde_ct_opt_weighted_fn(
                t_r, tsi_r, dsi_r, w_r, noise_sch, pointwise=True, keepdims=True
            )
            tilde_ct = tilde_ct.unsqueeze(1)   # weighted variant drops the MC axis
        else:
            _, tilde_ct = tilde_ct_opt_fn(
                t_r, tsi_r, dsi_r, noise_sch, pointwise=True, keepdims=True
            )
        # (B,1,n_p,1) -> average over particles -> (1,B,1)
        tilde_ct = tilde_ct.mean(dim=(-2, -1), keepdim=True).permute(1, 2, 0, 3).squeeze(0)
    else:
        t_r = t.unsqueeze(0).unsqueeze(-1)                             # (1,B,1)
        if weighted_stats:
            _, tilde_ct = tilde_ct_opt_weighted_fn(
                t_r, tsi_raw.unsqueeze(0), dsi_broad.unsqueeze(0), w.unsqueeze(0),
                noise_sch, pointwise=True, keepdims=True,
            )
        else:
            _, tilde_ct = tilde_ct_opt_fn(
                t_r, tsi_raw.unsqueeze(0), dsi_broad.unsqueeze(0),
                noise_sch, pointwise=True, keepdims=True,
            )
        tilde_ct = tilde_ct.squeeze(0)
    tilde_ct = torch.where(
        (tilde_ct.isnan() | tilde_ct.isinf()), torch.ones_like(tilde_ct), tilde_ct
    )

    cvsi_broad = (1 - tilde_ct) * tsi_broad + tilde_ct * dsi_broad
    score = (w * cvsi_broad).sum(dim=0)
    return (score, tilde_ct) if return_ct else score


def make_estimator(sampler: Callable, estimator: str = "cvsi", **estimator_kwargs):
    """Build ``fn(t, x, neg_energy_function, noise_sch, num_mc_samples)``.

    ``sampler`` is a callable ``(t, x, noise_sch, neg_energy_function, num_mc_samples) ->
    (x0_samples, w, h_t, log_w)``, e.g. ``mnist_ebm.make_rbm_gibbs_sampler(rbm)``.
    """

    def fn(t, x, neg_energy_function, noise_sch, num_mc_samples, **kw):
        x0, w, h_t, _ = sampler(t, x, noise_sch, neg_energy_function, num_mc_samples)
        return score_from_samples(
            t, x, x0, w, h_t, neg_energy_function, noise_sch,
            estimator=estimator, **{**estimator_kwargs, **kw},
        )

    return fn
