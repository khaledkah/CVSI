from __future__ import annotations

import os
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dem.models.components.cvsi import get_a2_b2

DEFAULT_MNIST_ROOT = ""


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_mnist(
    resolution: int = 14,
    digits: Optional[Iterable[int]] = None,
    n_max: Optional[int] = None,
    root: str = DEFAULT_MNIST_ROOT,
    train: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load MNIST, (optionally) downscale, and flatten to (N, D) in [0, 1].

    Args:
        resolution: Side length to bilinearly resize each image to (28 keeps the
            original). Lower resolutions keep the CPU-only MC sampling tractable.
        digits: If given, keep only these digit classes.
        n_max: If given, randomly subsample at most this many images.
        root: torchvision MNIST root (expects ``root/MNIST/raw/...``).
        train: Train vs. test split.

    Returns:
        (X, y) where X has shape (N, resolution**2) with pixel values in [0, 1].
    """
    import torchvision

    ds = torchvision.datasets.MNIST(root, train=train, download=True)
    X = ds.data.float() / 255.0  # (N, 28, 28)
    y = ds.targets.clone()

    if digits is not None:
        digits = list(digits)
        mask = torch.zeros(len(y), dtype=torch.bool)
        for d in digits:
            mask |= y == d
        X, y = X[mask], y[mask]

    if resolution != 28:
        X = F.interpolate(
            X.unsqueeze(1), size=(resolution, resolution),
            mode="bilinear", align_corners=False,
        ).squeeze(1)

    X = X.reshape(len(X), -1).contiguous()

    if n_max is not None and n_max < len(X):
        idx = torch.randperm(len(X))[:n_max]
        X, y = X[idx], y[idx]

    return X, y


class Standardizer:
    """Global (scalar) standardization, mirroring the GMM notebook convention.

    Diffusion noise schedules here assume the data has roughly unit variance, so
    we map pixels with a single global mean/std and keep the transform around to
    invert it for visualization.
    """

    def __init__(self, X: torch.Tensor):
        # Stored as plain floats (not tensors) so forward/inverse work on tensors
        # of any device (cpu or cuda) without a device-mismatch error.
        self.mean = X.mean().item()
        self.std = X.std().item()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.mean) / self.std

    def inverse(self, Z: torch.Tensor) -> torch.Tensor:
        return Z * self.std + self.mean


# ----------------------------------------------------------------------------
# Gaussian-Bernoulli RBM
# ----------------------------------------------------------------------------
class GaussianBernoulliRBM(nn.Module):
    """Continuous-visible / binary-hidden RBM with a closed-form free energy.

    The visible standard deviation ``sigma`` is fixed (data is pre-standardized to
    unit variance, so sigma = 1 is the natural choice).

    Free energy (energy of the marginal over visibles):
        F(v) = ||v - b||^2 / (2 sigma^2) - sum_j softplus(c_j + (W v)_j / sigma^2)
    """

    def __init__(self, n_visible: int, n_hidden: int = 256, sigma: float = 1.0):
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.register_buffer("sigma", torch.tensor(float(sigma)))
        # Small random init keeps the energy well-conditioned at the start.
        self.W = nn.Parameter(0.01 * torch.randn(n_hidden, n_visible))
        self.b = nn.Parameter(torch.zeros(n_visible))  # visible bias
        self.c = nn.Parameter(torch.zeros(n_hidden))   # hidden bias

    # --- core energy ---------------------------------------------------------
    def free_energy(self, v: torch.Tensor) -> torch.Tensor:
        """F(v), shape (...,) for input v of shape (..., D)."""
        s2 = self.sigma ** 2
        vbias = 0.5 * ((v - self.b) ** 2).sum(dim=-1) / s2
        pre_h = F.linear(v / s2, self.W, self.c)  # (..., H)
        hidden = F.softplus(pre_h).sum(dim=-1)
        return vbias - hidden

    def log_prob(self, v: torch.Tensor) -> torch.Tensor:
        """log p(v) up to an additive constant (= -F(v))."""
        return -self.free_energy(v)

    # --- Gibbs sampling (for training + visualization) -----------------------
    def h_given_v(self, v: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(F.linear(v / self.sigma ** 2, self.W, self.c))

    def v_given_h(self, h: torch.Tensor, sample: bool = False) -> torch.Tensor:
        mean = F.linear(h, self.W.t(), self.b)  # b + h W
        if sample:
            return mean + self.sigma * torch.randn_like(mean)
        return mean

    def gibbs_step(self, v: torch.Tensor, sample_v: bool = False) -> torch.Tensor:
        h = torch.bernoulli(self.h_given_v(v))
        return self.v_given_h(h, sample=sample_v)

    @torch.no_grad()
    def denoise(self, v: torch.Tensor) -> torch.Tensor:
        """Mean-field readout  E[v | h] = b + W^T p(h|v).

        A single deterministic reconstruction that projects a (possibly noisy)
        visible state onto the model's conditional mean. Because sigma = 1, the
        energy's minima tolerate ~1 std of per-pixel noise, so raw Gibbs / SDE
        draws look like static; this readout removes that visible-space noise
        while keeping the digit (the mode) the sample fell into. Use it for
        *visualization*, not as a substitute for real sampling.
        """
        return self.v_given_h(self.h_given_v(v), sample=False)

    @torch.no_grad()
    def posterior_gibbs(self, x_t: torch.Tensor, h_t: torch.Tensor,
                        n_steps: int = 100,
                        v_init: Optional[torch.Tensor] = None) -> torch.Tensor:
        """EXACT blocked Gibbs on the denoising posterior q(v | x_t).

        For VE diffusion (a(t)=1) the posterior is
            q(v | x_t) prop. to p(v) N(x_t; v, h_t I).
        Under a general kernel N(a(t) v, b^2(t) I) the same form is recovered by
        *whitening the observation*: p(v) N(x_t; a v, b^2 I) prop. to p(v) N(x_t/a; v,
        (b/a)^2 I), i.e. call this with ``x_t/a`` and ``h_t = b^2/a^2``. That is
        what :func:`make_rbm_gibbs_sampler` does, so this method itself needs no
        knowledge of the schedule.
        Reintroducing the RBM's binary hiddens h gives a joint
            q(v, h | x_t) prop. to exp(-E(v, h)) N(x_t; v, h_t I)
        that is *conjugate in both blocks*: the Gaussian likelihood term only
        touches v, so
            h | v        ~ Bernoulli(sigmoid(c + W v / sigma^2))   (unchanged), and
            v | h, x_t   ~ N(mu, 1/lam I)  with precision  lam = 1/sigma^2 + 1/h_t
                           and mean  mu = ((b + W^T h)/sigma^2 + x_t/h_t) / lam.
        Alternating the two blocks is exact MCMC targeting q(v | x_t): no importance
        weights, no step-size tuning.

        Args:
            x_t: noisy observation(s), shape (..., D).
            h_t: kernel variance b^2(t); scalar or broadcastable to (..., 1).
            v_init: chain initialization (defaults to x_t itself).
        """
        s2 = self.sigma ** 2
        h_t = torch.as_tensor(h_t, dtype=x_t.dtype, device=x_t.device)
        if h_t.ndim < x_t.ndim:
            h_t = h_t.reshape(h_t.shape + (1,) * (x_t.ndim - h_t.ndim))
        lam = 1.0 / s2 + 1.0 / h_t
        v = x_t.clone() if v_init is None else v_init
        for _ in range(n_steps):
            h = torch.bernoulli(self.h_given_v(v))
            mean = ((self.b + F.linear(h, self.W.t())) / s2 + x_t / h_t) / lam
            v = mean + torch.randn_like(mean) / lam.sqrt()
        return v

    @torch.no_grad()
    def sample(self, v_init: torch.Tensor, n_steps: int = 100,
               sample_v: bool = False, readout_mean: bool = False) -> torch.Tensor:
        """Run a Gibbs chain.

        Set ``sample_v=True`` so the chain actually mixes across modes (mean-field
        transitions have few fixed points and collapse); set ``readout_mean=True``
        to return the clean ``denoise`` reconstruction of the final state instead
        of the noisy visible sample.
        """
        v = v_init
        for _ in range(n_steps):
            v = self.gibbs_step(v, sample_v=sample_v)
        return self.denoise(v) if readout_mean else v


def train_rbm(
    rbm: GaussianBernoulliRBM,
    X: torch.Tensor,
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 1e-4,
    cd_k: int = 1,
    persistent: bool = True,
    num_fantasy: int = 256,
    sample_neg: bool = True,
    weight_decay: float = 1e-4,
    seed: int = 0,
    verbose: bool = True,
) -> list:
    """Train the RBM with (persistent) contrastive divergence.

    The CD loss is  mean F(v_data) - mean F(v_model), where ``v_model`` comes from
    k Gibbs steps of a Markov chain. Autograd then supplies the standard RBM
    parameter gradients (data statistics minus model statistics).

    Two choices here matter a lot for avoiding **mode collapse** on MNIST (an
    under-trained GB-RBM otherwise concentrates on the ink-mean blob, which reads
    as a single "0"):

    - ``persistent`` keeps a **fixed-size pool of fantasy particles** (size
      ``num_fantasy``) that evolves continuously across *all* minibatches and
      epochs. Crucially the pool is never reset to the current minibatch, so the
      negative chain mixes over the whole model distribution instead of being
      repeatedly yanked back to data.
    - ``sample_neg`` draws **stochastic visibles** in the negative phase
      (``v ~ N(b + W^T h, sigma^2 I)`` rather than its mean). Mean-field negatives
      never expose the model to visible-space noise, so it under-learns variance
      and collapses; sampling visibles is what keeps the learned energy multimodal.

    Returns the per-epoch list of mean data free energies (a rough training curve).
    """
    torch.manual_seed(seed)
    opt = torch.optim.Adam(rbm.parameters(), lr=lr, weight_decay=weight_decay)
    N = len(X)
    history = []

    # Persistent pool of fantasy particles, seeded once from random data points
    # and thereafter evolved by Gibbs sampling only (never reset to data).
    fantasy = None
    if persistent:
        fantasy = X[torch.randperm(N, device=X.device)[:num_fantasy]].clone()

    for ep in range(epochs):
        perm = torch.randperm(N, device=X.device)
        running = 0.0
        nb = 0
        for i in range(0, N, batch_size):
            v_data = X[perm[i:i + batch_size]]

            v_model = fantasy if persistent else v_data
            for _ in range(cd_k):
                v_model = rbm.gibbs_step(v_model, sample_v=sample_neg)
            v_model = v_model.detach()
            if persistent:
                fantasy = v_model

            loss = rbm.free_energy(v_data).mean() - rbm.free_energy(v_model).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rbm.parameters(), 10.0)
            opt.step()

            running += rbm.free_energy(v_data).mean().item()
            nb += 1

        history.append(running / max(nb, 1))
        if verbose:
            print(f"epoch {ep + 1:3d}/{epochs}  mean F(data) = {history[-1]:+.4f}")

    return history


def make_rbm_gibbs_sampler(rbm: "GaussianBernoulliRBM", n_steps: int = 100):
    """Adapter: exact RBM posterior Gibbs in the ``estimators`` interface.

    Returns a callable ``(t, x, noise_sch, neg_energy_function, num_mc_samples)
    -> (x0_samples, w, h_t, log_w)`` with uniform weights (exact MCMC needs no
    reweighting), usable with ``estimators.make_estimator(sampler=...)``.

    Schedule-general (VE *and* VP), like ``estimators.score_from_samples``: the
    kernel is ``N(a(t) x0, b^2(t) I)`` via ``cvsi.get_a2_b2``, and the posterior
    ``p(x0) N(x_t; a x0, b^2 I) prop. to p(x0) N(x_t/a; x0, (b/a)^2 I)`` is sampled by
    handing ``posterior_gibbs`` the whitened observation ``x_t/a`` with variance
    ``b^2/a^2``. The returned ``h_t`` is ``b^2(t)``, the kernel variance the other
    samplers return and the DSI denominator.
    """

    def sampler(t, x, noise_sch, neg_energy_function, num_mc_samples):
        if t.ndim == 0:
            t = t.unsqueeze(0).repeat(len(x))
        repeated_t = t.unsqueeze(0).repeat_interleave(num_mc_samples, dim=0)
        repeated_x = x.unsqueeze(0).repeat_interleave(num_mc_samples, dim=0)
        a2_t, b2_t = get_a2_b2(repeated_t, noise_sch)
        a_t = a2_t.sqrt().unsqueeze(-1)          # (K,B,1)
        h_t = b2_t.unsqueeze(-1)                 # (K,B,1) = b^2(t); == h(t) under VE
        # Whitened observation: the posterior in x0 is N(x_t/a, (b/a)^2 I) tilted by p.
        y_t = repeated_x / a_t
        h_eff = h_t / a_t.pow(2)                 # (b/a)^2
        v0 = y_t + torch.randn_like(y_t) * h_eff.sqrt()
        v = rbm.posterior_gibbs(y_t, h_eff, n_steps=n_steps, v_init=v0)
        log_w = neg_energy_function(v).reshape(h_t.shape[:-1])
        w = torch.full_like(h_t, 1.0 / num_mc_samples)
        return v, w, h_t, log_w

    return sampler


# ----------------------------------------------------------------------------
# Energy wrapper for the CVSI / TSI / DSI estimators
# ----------------------------------------------------------------------------
class LogProbEnergy:
    """Adapter so an RBM looks like the energy objects the estimators expect.

    The estimators in ``dem.models.components.cvsi`` call
    ``neg_energy_function(x0)`` expecting **log p(x0)** (a.k.a. the negative
    energy), differentiate it with ``torch.func.grad``, and occasionally read an
    ``is_molecule`` flag. This thin wrapper provides exactly that interface.
    """

    is_molecule = False

    def __init__(self, rbm: GaussianBernoulliRBM):
        self.rbm = rbm

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        return self.rbm.log_prob(v)
