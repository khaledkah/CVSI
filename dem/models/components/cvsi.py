from typing import Callable, Tuple, Optional

import numpy as np
import torch
from .clipper import Clipper
from dem.utils.data_utils import remove_mean

# Log-weight floor used to exclude proposal samples with non-finite energies from
# the self-normalized IS sum. If every sample of a batch element is non-finite,
# softmax over a constant yields uniform weights (finite), and the tilde_ct -> 1
# fallback reduces the estimate to pure DSI, which is always finite.
_LOG_W_FLOOR = -1e30

# Rounds of (per-particle norm clip -> mean-free projection) for molecular scores.
# See clip_scores: one round alone breaks either the norm bound or mean-freeness.
_MOL_CLIP_PROJECT_ROUNDS = 3


def get_a2_b2(
    t_steps: torch.Tensor,
    noise_sch: Callable[[torch.Tensor], torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes a^2(t) and b^2(t) of the perturbation kernel N(a(t) x0, b^2(t) I).

    A schedule exposing ``a2_b2(t)`` (the VP / SNR-TV schedule) provides its own
    (a^2, b^2); otherwise the VE contract is used (a(t)=1, b^2(t)=h(t)).

    Args:
        t_steps: Time steps tensor.
    Returns:
        Tuple of (a2, b2) tensors.
    """
    if hasattr(noise_sch, "a2_b2"):
        return noise_sch.a2_b2(t_steps)
    b2_t = noise_sch.h(t_steps)
    a2_t = torch.ones_like(b2_t)
    return a2_t, b2_t


def cov_multi_dim(
    x: torch.Tensor,
    y: torch.Tensor,
    pointwise: bool = False,
    divide_by_dim: bool = False,
) -> torch.Tensor:
    """Computes covariance between batches of multichannel data.

    Args:
      x: First tensor, shape (..., num_channels, sequence_length).
      y: Second tensor, shape (..., num_channels, sequence_length).
      pointwise: If False, average covariance over channels.
      divide_by_dim: If True, divide covariance by sequence_length.

    Returns:
      Covariance tensor.
    """
    x_mean = x.mean(dim=1, keepdim=True)
    y_mean = y.mean(dim=1, keepdim=True)

    cov = ((x - x_mean) * (y - y_mean)).sum(dim=-1).mean(dim=-2)

    if divide_by_dim:
        cov = cov / x.shape[-1]

    if not pointwise:
        cov = cov.mean(-1)

    return cov


def cov_multi_dim_mult(
    x: torch.Tensor, y: torch.Tensor, pointwise: bool = False
) -> torch.Tensor:
    """Computes covariance between batches of multichannel data.

    Args:
      x: First tensor, shape (..., num_channels, sequence_length).
      y: Second tensor, shape (..., num_channels, sequence_length).
      pointwise: If False, average covariance over channels.

    Returns:
      Covariance tensor.
    """
    x_mean = x.mean(dim=1, keepdim=True)
    y_mean = y.mean(dim=1, keepdim=True)

    cov = ((x - x_mean) * (y - y_mean)).mean(dim=-3)

    if not pointwise:
        cov = cov.mean(-2)

    return cov


def tilde_ct_opt_fn(
    t_steps: torch.Tensor,
    tsi_broad_scores: torch.Tensor,
    dsi_broad_scores: torch.Tensor,
    noise_sch: Callable[[torch.Tensor], torch.Tensor],
    pointwise: bool,
    keepdims: bool = True,
    divide_by_dim: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes optimal control variate coefficient c*(t) and c*(t)a(t).

    Args:
      t_steps: Time steps.
      tsi_broad_scores: TSI scores with MC sample dimension.
      dsi_broad_scores: DSI scores with MC sample dimension.
    noise_sch: Noise schedule object (provides h(t)).
      pointwise: If False, average variance/covariance over channels.
      keepdims: If True, keep squeezed dimensions.
      divide_by_dim: If True, divide variance/covariance by dimension.

    Returns:
      Tuple of (ct_opt, tilde_ct_opt).
    """
    t_steps = t_steps.unsqueeze(-3)

    a2_t, _ = get_a2_b2(t_steps, noise_sch)
    a_t = torch.sqrt(a2_t)
    scaled_tsi_broad_scores = a_t * tsi_broad_scores

    # unbiased=False so the variances use the same 1/M normalization as
    # cov_multi_dim below; otherwise c* is systematically distorted by
    # (M-1)/M.
    var_scaled_tsi = scaled_tsi_broad_scores.var(
        dim=(-3), unbiased=False, keepdim=True
    ).sum(dim=-1, keepdim=True)  # \tilde{g}
    var_dsi = dsi_broad_scores.var(dim=(-3), unbiased=False, keepdim=True).sum(
        dim=-1, keepdim=True
    )

    if divide_by_dim:
        var_scaled_tsi = var_scaled_tsi / dsi_broad_scores.shape[-1]
        var_dsi = var_dsi / dsi_broad_scores.shape[-1]

    cov_scaled_tsi_dsi = cov_multi_dim(
        scaled_tsi_broad_scores,
        dsi_broad_scores,
        pointwise=pointwise,
        divide_by_dim=divide_by_dim,
    )

    # print(cov_scaled_tsi_dsi)

    if not pointwise:
        var_scaled_tsi = var_scaled_tsi.mean(dim=-2, keepdim=True)
        var_dsi = var_dsi.mean(dim=-2, keepdim=True)

    cov_scaled_tsi_dsi = cov_scaled_tsi_dsi.reshape(var_dsi.shape)

    ct_opt = (var_scaled_tsi - a_t * cov_scaled_tsi_dsi) / (
        a_t * var_scaled_tsi + a_t**3 * var_dsi - 2 * a_t**2 * cov_scaled_tsi_dsi
    )
    tilde_ct_opt = a_t * ct_opt

    if not keepdims:
        ct_opt = ct_opt.squeeze()
        tilde_ct_opt = tilde_ct_opt.squeeze()

    return ct_opt, tilde_ct_opt


###############################
# Simplified iDEM CVSI (proposal only, single energy eval)
###############################


def tilde_ct_opt_weighted_fn(
    t_steps: torch.Tensor,
    tsi_broad_scores: torch.Tensor,
    dsi_broad_scores: torch.Tensor,
    w: torch.Tensor,
    noise_sch: Callable[[torch.Tensor], torch.Tensor],
    pointwise: bool,
    keepdims: bool = True,
    divide_by_dim: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes optimal control variate coefficient c*(t) and c*(t)a(t).

    Uses importance sampling weights to compute variances and covariances
    under the target distribution q(x0|xt), rather than the proposal.

    Args:
        t_steps: Time steps.
        tsi_broad_scores: TSI scores with MC sample dimension (M,B,D).
        dsi_broad_scores: DSI scores with MC sample dimension (M,B,D).
        w: Importance sampling weights (M,B,1), normalized over MC axis.
        noise_sch: Noise schedule object (provides h(t)).
        pointwise: If False, average variance/covariance over channels.
        keepdims: If True, keep squeezed dimensions.
        divide_by_dim: If True, divide variance/covariance by dimension.

    Returns:
        Tuple of (ct_opt, tilde_ct_opt).
    """
    t_steps = t_steps.unsqueeze(-3)

    a2_t, _ = get_a2_b2(t_steps, noise_sch)
    a_t = torch.sqrt(a2_t)
    scaled_tsi_broad_scores = a_t * tsi_broad_scores

    # Weighted mean over MC dimension (dim=1 after unsqueeze(0))
    weighted_mean_scaled_tsi = (w * scaled_tsi_broad_scores).sum(dim=1)
    weighted_mean_dsi = (w * dsi_broad_scores).sum(dim=1)

    # Weighted variance
    diff_scaled_tsi = scaled_tsi_broad_scores - weighted_mean_scaled_tsi.unsqueeze(1)
    diff_dsi = dsi_broad_scores - weighted_mean_dsi.unsqueeze(1)

    var_scaled_tsi = (w * diff_scaled_tsi**2).sum(dim=1).sum(dim=-1, keepdim=True)
    var_dsi = (w * diff_dsi**2).sum(dim=1).sum(dim=-1, keepdim=True)

    # Weighted covariance
    cov_scaled_tsi_dsi = (
        (w * diff_scaled_tsi * diff_dsi).sum(dim=1).sum(dim=-1, keepdim=True)
    )

    if divide_by_dim:
        var_scaled_tsi = var_scaled_tsi / dsi_broad_scores.shape[-1]
        var_dsi = var_dsi / dsi_broad_scores.shape[-1]
        cov_scaled_tsi_dsi = cov_scaled_tsi_dsi / dsi_broad_scores.shape[-1]

    if not pointwise:
        var_scaled_tsi = var_scaled_tsi.mean(dim=-2, keepdim=True)
        var_dsi = var_dsi.mean(dim=-2, keepdim=True)

    # Weighted average of a_t over MC dimension
    a_t_weighted = (w * a_t).sum(dim=1)

    ct_opt = (var_scaled_tsi - a_t_weighted * cov_scaled_tsi_dsi) / (
        a_t_weighted * var_scaled_tsi
        + a_t_weighted**3 * var_dsi
        - 2 * a_t_weighted**2 * cov_scaled_tsi_dsi
        + 1e-12
    )
    tilde_ct_opt = a_t_weighted * ct_opt

    if not keepdims:
        ct_opt = ct_opt.squeeze()
        tilde_ct_opt = tilde_ct_opt.squeeze()

    return ct_opt, tilde_ct_opt


def sample_proposal_x0_and_weights(
    t: torch.Tensor,
    x: torch.Tensor,
    noise_sch: Callable[[torch.Tensor], torch.Tensor],
    neg_energy_function: Callable[[torch.Tensor], torch.Tensor],
    num_mc_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample proposal x0 ~ N(xt, h_t I) and compute log weights & softmax weights.

    Single evaluation of neg_energy_function used for both weights and TSI gradient.
    Returns:
        x0_samples: (M,B,D)
        w: (M,B,1) softmax-normalized over MC axis (proposal weighting for aggregation)
        h_t: (M,B,1) variance parameter (b^2(t))
        log_w: (M,B) = log p(x0)
    """
    if t.ndim == 0:
        t = t.unsqueeze(0).repeat(len(x))
    repeated_t = t.unsqueeze(0).repeat_interleave(num_mc_samples, dim=0)
    repeated_x = x.unsqueeze(0).repeat_interleave(num_mc_samples, dim=0)
    # For VP (a(t)!=1) the optimal proposal for q(x0|xt) ~ p(x0) N(xt; a x0, b^2)
    # is N(xt/a, (b/a)^2 I); h_t below carries b^2(t) (the DSI denominator). For
    # VE (a2=1, b2=h) this reduces exactly to the old N(xt, h I) proposal.
    a2_t, b2_t = get_a2_b2(repeated_t, noise_sch)
    a_t = a2_t.sqrt().unsqueeze(-1)
    h_t = b2_t.unsqueeze(-1)  # = b^2(t); == h(t) under VE
    prop_mean = repeated_x / a_t
    prop_std = (h_t / a_t.pow(2)).sqrt()  # b/a

    noise = torch.randn_like(repeated_x)
    if getattr(neg_energy_function, "is_molecule", False):
        # For mean-free (translation-invariant) systems the diffusion lives on
        # the mean-free subspace, so the proposal noise must too. Otherwise the
        # DSI term (a x0 - xt)/b^2 carries a pure center-of-mass noise component
        # that the true score cannot have, which inflates Var(DSI) in the
        # tilde_ct statistics and adds unlearnable noise to the target.
        noise = remove_mean(
            noise,
            neg_energy_function.n_particles,
            neg_energy_function.n_spatial_dim,
        )
    x0_samples = prop_mean + noise * prop_std

    # Value-only eval for the IS weights, under no_grad; the TSI gradient is taken
    # separately via torch.func.grad, which works correctly under it.
    with torch.no_grad():
        log_w = neg_energy_function(x0_samples)

    log_w = log_w.reshape(h_t.shape)

    # Exclude proposal samples with non-finite energies (e.g. overlapping
    # particles early in training) from the IS sum: they get weight ~0. If all
    # samples of a batch element are non-finite, softmax over the constant
    # floor yields uniform weights instead of NaN.
    log_w = torch.nan_to_num(
        log_w, nan=_LOG_W_FLOOR, posinf=_LOG_W_FLOOR, neginf=_LOG_W_FLOOR
    )

    w = torch.softmax(log_w, dim=0)

    return x0_samples, w, h_t, log_w


def clip_scores(
    scores: torch.Tensor, clipper: Clipper, neg_energy_function
) -> torch.Tensor:
    # Norm-clipping cannot fix non-finite scores, so zero them out instead.
    # Zero (rather than a large constant) avoids injecting a spurious
    # fixed-direction target: combined with the sanitized IS weights, a bad
    # sample then contributes nothing to the estimate.
    scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    if neg_energy_function.is_molecule:
        old_shape = scores.shape
        new_shape = scores.shape[:-1] + (
            neg_energy_function.n_particles,
            neg_energy_function.n_spatial_dim,
        )

        scores = scores.reshape(new_shape)

        # Clip and re-project alternately. grad log p of a translation-invariant
        # energy is exactly mean-free, but the clip scales each particle by a
        # different coefficient min(1, max_norm/||g_i||) and destroys that. The COM
        # component it introduces is not part of any valid score. Projecting alone
        # would let a particle exceed max_norm again (||g_i - mean|| can grow), so
        # the two alternate; a few rounds converge.
        for _ in range(_MOL_CLIP_PROJECT_ROUNDS):
            scores = clipper.clip_scores(scores)
            scores = scores - scores.mean(dim=-2, keepdim=True)

        scores = scores.reshape(old_shape)
        return scores

    scores = clipper.clip_scores(scores)

    return scores


def idem_score_cvsi_fn(
    t: torch.Tensor,
    x: torch.Tensor,
    neg_energy_function: Callable[[torch.Tensor], torch.Tensor],
    noise_sch: Callable[[torch.Tensor], torch.Tensor],
    num_mc_samples: int,
    debug: bool = False,
    pointwise: bool = False,
    divide_by_dim: bool = False,
    weighted_stats: bool = False,
    ct_per_molecule: bool = False,
    clipper_tsi: Optional[Clipper] = None,
) -> torch.Tensor:
    """CVSI using proposal samples with optional weighted statistics.

    Steps:
      1. Sample x0 from proposal N(xt, h_t I).
      2. Compute log_w = log p(x0) once; use for weights and TSI gradient.
      3. Form DSI and TSI broad scores.
      4. Compute tilde c_t via tilde_ct_opt_fn (unweighted) or
         tilde_ct_opt_weighted_fn (weighted).
      5. Combine scores, aggregate with softmax weights.

    Args:
        t: Time tensor (B,) or scalar.
        x: Observation tensor (B, D).
        neg_energy_function: Returns log p(x0).
        noise_sch: Noise schedule with h(t) method.
        num_mc_samples: Number of MC samples.
        pointwise: If True, return per-sample weighted scores.
        divide_by_dim: If True, divide variance/covariance by dimension.
        weighted_stats: If True, use IS weights in computing tilde c_t.
        ct_per_molecule: If True, compute tilde c_t per particle for molecular
            energies, then average.
        clipper_tsi: Norm clip on the TSI branch mixed into the returned score. The
            tilde c_t statistics always use the unclipped TSI.

    Returns:
        If pointwise=False: (B, D) aggregated CVSI scores.
        If pointwise=True: tuple of (weighted_cvsi, w, tilde_ct).
    """
    # Sampling & weights (single energy eval)
    x0_samples, w, h_t, log_w = sample_proposal_x0_and_weights(
        t, x, noise_sch, neg_energy_function, num_mc_samples
    )

    # Call under no_grad: torch.func.grad otherwise retains a graph through the
    # energy's own parameters. The gradient w.r.t. x0_samples is correct either way.
    grad_fn = torch.func.grad(lambda x0: neg_energy_function(x0).sum())
    with torch.no_grad():
        tsi_broad = grad_fn(x0_samples)

    # Zero out non-finite gradients (exploding energies) before they enter the
    # tilde_ct statistics or the weighted sum: the corresponding samples
    # already received weight ~0 above, but 0 * NaN would still be NaN.
    tsi_broad = torch.nan_to_num(tsi_broad, nan=0.0, posinf=0.0, neginf=0.0)

    # Clip the TSI scores that are mixed into the returned score, so the regression
    # label stays bounded on exploding-energy regions; tilde_ct is computed from the
    # unclipped branch. DSI is never clipped.
    tsi_for_stats = tsi_broad
    if clipper_tsi is not None:
        tsi_broad = clip_scores(tsi_broad, clipper_tsi, neg_energy_function)

    # DSI (a(t)=1, b^2(t)=h_t)
    dsi_broad = (x0_samples.detach() - x.unsqueeze(0)) / h_t

    if ct_per_molecule and neg_energy_function.is_molecule:
        tsi_broad_reshaped = tsi_for_stats.reshape(
            tsi_for_stats.shape[0],
            tsi_for_stats.shape[1],
            neg_energy_function.n_particles,
            neg_energy_function.n_spatial_dim,
        ).permute(1, 0, 2, 3)
        dsi_broad_reshaped = dsi_broad.reshape(
            dsi_broad.shape[0],
            dsi_broad.shape[1],
            neg_energy_function.n_particles,
            neg_energy_function.n_spatial_dim,
        ).permute(1, 0, 2, 3)
        w_reshaped = w.reshape(
            w.shape[0],
            w.shape[1],
            1,
            1,
        ).permute(1, 0, 2, 3)
        t = t.unsqueeze(-1).unsqueeze(-1)
    else:
        tsi_broad_reshaped = tsi_for_stats.unsqueeze(0)
        dsi_broad_reshaped = dsi_broad.unsqueeze(0)
        t = t.unsqueeze(0).unsqueeze(-1)
        w_reshaped = w.unsqueeze(0)

    # Control variate coefficient
    if weighted_stats:
        _, tilde_ct = tilde_ct_opt_weighted_fn(
            t,
            tsi_broad_reshaped,
            dsi_broad_reshaped,
            w_reshaped,
            noise_sch,
            pointwise=True,
            keepdims=True,
            divide_by_dim=divide_by_dim,
        )
    else:
        _, tilde_ct = tilde_ct_opt_fn(
            t,
            tsi_broad_reshaped,
            dsi_broad_reshaped,
            noise_sch,
            pointwise=True,
            keepdims=True,
            divide_by_dim=divide_by_dim,
        )
    if ct_per_molecule and neg_energy_function.is_molecule:
        if weighted_stats:
            tilde_ct = tilde_ct.unsqueeze(1)
        tilde_ct = tilde_ct.mean(dim=(-2, -1), keepdim=True).permute(1, 2, 0, 3)
    tilde_ct = tilde_ct.squeeze(0)

    # replace nans in ct with 1.0 (equivalent to using only DSI, because of exploding TSI)
    tilde_ct = torch.where(
        (tilde_ct.isnan() | tilde_ct.isinf()), torch.ones_like(tilde_ct), tilde_ct
    )

    cvsi_broad = (1 - tilde_ct) * tsi_broad + tilde_ct * dsi_broad

    weighted_cvsi = w * cvsi_broad

    if not pointwise:
        if debug:
            return (
                weighted_cvsi.sum(dim=0),
                w,
                tilde_ct,
                tsi_broad,
                dsi_broad,
                cvsi_broad,
                log_w,
            )
        else:
            return weighted_cvsi.sum(dim=0)
    else:
        return weighted_cvsi, w, tilde_ct
