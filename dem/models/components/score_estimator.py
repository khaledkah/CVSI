import numpy as np
import torch

from dem.energies.base_energy_function import BaseEnergyFunction
from dem.models.components.clipper import Clipper
from dem.models.components.noise_schedules import BaseNoiseSchedule


def wrap_for_richardsons(score_estimator):
    def _fxn(t, x, energy_function, noise_schedule, num_mc_samples):
        bigger_samples = score_estimator(
            t, x, energy_function, noise_schedule, num_mc_samples
        )

        smaller_samples = score_estimator(
            t, x, energy_function, noise_schedule, int(num_mc_samples / 2)
        )

        return (2 * bigger_samples) - smaller_samples

    return _fxn


def log_expectation_reward(
    t: torch.Tensor,
    x: torch.Tensor,
    energy_function: BaseEnergyFunction,
    noise_schedule: BaseNoiseSchedule,
    num_mc_samples: int,
    clipper: Clipper = None,
):
    repeated_t = t.unsqueeze(0).repeat_interleave(num_mc_samples, dim=0)
    repeated_x = x.unsqueeze(0).repeat_interleave(num_mc_samples, dim=0)

    h_t = noise_schedule.h(repeated_t).unsqueeze(1)

    samples = repeated_x + (torch.randn_like(repeated_x) * h_t.sqrt())

    log_rewards = energy_function(samples)

    if clipper is not None and clipper.should_clip_log_rewards:
        log_rewards = clipper.clip_log_rewards(log_rewards)

    return torch.logsumexp(log_rewards, dim=-1) - np.log(num_mc_samples)


def estimate_grad_Rt(
    t: torch.Tensor,
    x: torch.Tensor,
    energy_function: BaseEnergyFunction,
    noise_schedule: BaseNoiseSchedule,
    num_mc_samples: int,
):
    if t.ndim == 0:
        t = t.unsqueeze(0).repeat(len(x))

    grad_fxn = torch.func.grad(log_expectation_reward, argnums=1)
    vmapped_fxn = torch.vmap(
        grad_fxn, in_dims=(0, 0, None, None, None), randomness="different"
    )

    return vmapped_fxn(t, x, energy_function, noise_schedule, num_mc_samples)


def log_expectation_reward_batch(
    t: torch.Tensor,
    x: torch.Tensor,
    energy_function: BaseEnergyFunction,
    noise_schedule: BaseNoiseSchedule,
    num_mc_samples: int,
    clipper: Clipper = None,
):
    """Vectorized (batch) version of log_expectation_reward.

    Computes log E_{q(x0|xt)}[p(x0)] for an entire batch of (t, x) pairs simultaneously
    without relying on torch.vmap by building a single MC sample tensor.

    Args:
        t: Times tensor of shape (B,) or (B, 1, ...).
        x: Positions tensor of shape (B, D).
        energy_function: Callable returning log p(x) given inputs of shape (N, D).
        noise_schedule: Provides h(t).
        num_mc_samples: Number of Monte Carlo samples.
        clipper: Optional Clipper for log reward stabilization.

    Returns:
        Tensor of shape (B,) with log expectation rewards per sample.
    """
    if t.ndim > 1:
        t = t.view(t.shape[0])  # squeeze extra dims if present

    repeated_t = t.unsqueeze(0).repeat(num_mc_samples, 1)
    repeated_x = x.unsqueeze(0).repeat(num_mc_samples, 1, 1)

    h_t = noise_schedule.h(repeated_t).unsqueeze(-1)

    samples = repeated_x + torch.randn_like(repeated_x) * h_t.sqrt()

    log_rewards = energy_function(samples)

    if clipper is not None and clipper.should_clip_log_rewards:
        log_rewards = clipper.clip_log_rewards(log_rewards)

    # log-sum-exp over MC samples -> (B,)
    return torch.logsumexp(log_rewards, dim=0) - np.log(num_mc_samples)


def estimate_grad_Rt_parallel(
    t: torch.Tensor,
    x: torch.Tensor,
    energy_function: BaseEnergyFunction,
    noise_schedule: BaseNoiseSchedule,
    num_mc_samples: int,
):
    """Parallel (batched) gradient estimator of R_t without torch.vmap.

    Uses a single Monte Carlo tensor of shape (M, B, D) and computes gradients
    for all batch elements simultaneously. Since each element's reward depends
    only on its own row of x, taking the gradient of the summed rewards yields
    the per-sample gradients.

    Args:
        t: Times tensor of shape (B,) or (B,1,...).
        x: Input positions (B, D).
        energy_function: Callable returning log p(x).
        noise_schedule: Provides h(t).
        num_mc_samples: Monte Carlo samples per element.

    Returns:
        Gradient estimates with shape (B, D).
    """
    if t.ndim == 0:
        t = t.unsqueeze(0).repeat(x.shape[0])

    x.requires_grad_(True)
    energy_sum = energy_function(x).sum()
    scores = torch.autograd.grad(energy_sum, x, create_graph=False)[0].detach()
    x.requires_grad_(False)

    return scores
