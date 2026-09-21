from abc import ABC, abstractmethod

import numpy as np
import torch


class BaseNoiseSchedule(ABC):
    @abstractmethod
    def g(t):
        # Returns g(t)
        pass

    @abstractmethod
    def h(t):
        # Returns \int_0^t g(t)^2 dt
        pass


class LinearNoiseSchedule(BaseNoiseSchedule):
    def __init__(self, beta, min_t=1e-5, name="linear"):
        self.beta = beta
        self.min_t = min_t
        self.name = name

    def g(self, t):
        t = torch.clamp(t, min=self.min_t)
        return torch.full_like(t, self.beta**0.5)

    def h(self, t):
        t = torch.clamp(t, min=self.min_t)
        return self.beta * t


class QuadraticNoiseSchedule(BaseNoiseSchedule):
    def __init__(self, beta, min_t=1e-5, name="quadratic"):
        self.beta = beta
        self.min_t = min_t
        self.name = name

    def g(self, t):
        t = torch.clamp(t, min=self.min_t)
        return torch.sqrt(self.beta * 2 * t)

    def h(self, t):
        t = torch.clamp(t, min=self.min_t)
        return self.beta * t**2


class PowerNoiseSchedule(BaseNoiseSchedule):
    def __init__(self, beta, power, min_t=1e-5, name="power"):
        self.beta = beta
        self.power = power
        self.min_t = min_t
        self.name = name

    def g(self, t):
        t = torch.clamp(t, min=self.min_t)
        return torch.sqrt(self.beta * self.power * (t ** (self.power - 1)))

    def h(self, t):
        t = torch.clamp(t, min=self.min_t)
        return self.beta * (t**self.power)


class SubLinearNoiseSchedule(BaseNoiseSchedule):
    def __init__(self, beta, min_t=1e-5, name="sub_linear"):
        self.beta = beta
        self.min_t = min_t
        self.name = name

    def g(self, t):
        t = torch.clamp(t, min=self.min_t)
        return torch.sqrt(self.beta * 0.5 * 1 / (t**0.5 + 1e-3))

    def h(self, t):
        t = torch.clamp(t, min=self.min_t)
        return self.beta * t**0.5


class GeometricNoiseSchedule(BaseNoiseSchedule):
    def __init__(self, sigma_min, sigma_max, min_t=1e-5, name="geometric"):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_diff = self.sigma_max / self.sigma_min
        self.min_t = min_t
        self.name = name

    def g(self, t):
        t = torch.clamp(t, min=self.min_t)
        # Let sigma_d = sigma_max / sigma_min
        # Then g(t) = sigma_min * sigma_d^t * sqrt{2 * log(sigma_d)}
        # See Eq 192 in https://arxiv.org/pdf/2206.00364.pdf
        return (
            self.sigma_min
            * (self.sigma_diff**t)
            * ((2 * np.log(self.sigma_diff)) ** 0.5)
        )

    def h(self, t):
        t = torch.clamp(t, min=self.min_t)
        # Let sigma_d = sigma_max / sigma_min
        # Then h(t) = \int_0^t g(z)^2 dz = sigma_min * sqrt{sigma_d^{2t} - 1}
        # see Eq 199 in https://arxiv.org/pdf/2206.00364.pdf
        return (self.sigma_min * (((self.sigma_diff ** (2 * t)) - 1) ** 0.5)) ** 2


class KVeNoiseSchedule(BaseNoiseSchedule):
    """
    Karras et al. 2022 Variance-Exploding (VE) noise schedule with rho-parameterization.
    """

    def __init__(
        self,
        sigma_min: float,
        sigma_max: float,
        rho: float = 7.0,
        name: str = "kve",
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.name = name
        self.smin_r = self.sigma_min ** (1.0 / self.rho)
        self.smax_r = self.sigma_max ** (1.0 / self.rho)

    def _base(self, t: torch.Tensor) -> torch.Tensor:
        return self.smax_r + (1.0 - t) * (self.smin_r - self.smax_r)

    def _sigma(self, t: torch.Tensor) -> torch.Tensor:
        return torch.pow(self._base(t), self.rho)

    def g(self, t: torch.Tensor) -> torch.Tensor:
        # g(t)^2 = d/dt [sigma(t)^2] = 2 * sigma(t) * sigma'(t)
        sigma_t = self._sigma(t)
        sigma_prime = (
            self.rho
            * torch.pow(self._base(t), self.rho - 1)
            * (self.smax_r - self.smin_r)
        )
        g_sq = 2.0 * sigma_t * sigma_prime
        return torch.sqrt(g_sq)

    def h(self, t: torch.Tensor) -> torch.Tensor:
        # Intentionally sigma(t)^2 rather than the base-class contract
        # int_0^t g^2 = sigma(t)^2 - sigma_min^2: the sigma_min^2 floor at
        # t=0 keeps h(t) bounded away from zero, which is needed for
        # numerical stability (h appears in denominators of DSI and the
        # proposal variance). h'(t) = g(t)^2 still holds, so the schedule
        # remains internally consistent with the reverse SDE.
        sigma_t = self._sigma(t)
        return torch.pow(sigma_t, 2.0)


# ---------------------------------------------------------------------------
# VP / SNR-TV schedule (https://arxiv.org/abs/2502.08598). Kernel is
# q(x_t|x_0)=N(a(t) x0, b(t)^2 I) with a(t)!=1, parameterised by an SNR schedule
# gamma^2(t) and a total-variance (TV) schedule tau^2(t):
#   a^2 = tau^2 gamma^2/(gamma^2+1),  b^2 = a^2/gamma^2.
# It exposes an `a2_b2(t)` method; `cvsi.get_a2_b2` and the reverse SDE dispatch on
# `hasattr(sched, "a2_b2")` / `hasattr(sched, "f_drift")`.
# ---------------------------------------------------------------------------
def exp_inverse_sigmoid_snr_sch(t, slope: float, shift: float):
    """ISSNR schedule (https://arxiv.org/abs/2502.08598): gamma^2 = (1/t - 1)^slope * e^shift."""
    log_gamma2_t = torch.log(1.0 / t - 1.0) * slope + shift
    return torch.exp(log_gamma2_t)


def constant_tv_sch(t, scale: float = 1.0):
    """Constant total-variance schedule tau^2(t) = scale (variance-preserving)."""
    return torch.full_like(t, float(scale))


def fm_tv_sch(t, slope: float, shift: float):
    """Flow-matching TV schedule tau^2 = (1-t)^slope + t^slope e^{-shift}.

    Only meaningful *paired with ISSNR of the same (slope, shift)*, in which case
    gamma^2 = ((1-t)/t)^slope e^shift and the algebra collapses to

        a^2(t) = (1-t)^slope,   b^2(t) = t^slope e^{-shift},

    i.e. the flow-matching / stochastic-interpolant kernel N((1-t)^{s/2} x0, t^s e^{-shift} I);
    with slope=2, shift=0 this is exactly the linear interpolant a=1-t, b=t. Unlike
    constant TV the total variance is *not* preserved (tau^2 dips to its minimum
    mid-trajectory: 2^{1-slope} at slope>1, shift=0), so the marginal at the reverse-SDE
    start depends on the start time -- use ``prior_std(start)``, not ``prior_std()``.
    ``SNRTVNoiseSchedule(tv="fm")`` wires the pairing up for you.
    """
    return (1.0 - t) ** slope + t**slope * float(np.exp(-shift))


def _log_deriv(sch, t):
    """d/dt [0.5 * log sch(t)], elementwise, via autograd.

    ``allow_unused`` so a schedule constant in t (e.g. constant_tv_sch) yields 0
    rather than erroring. ``torch.enable_grad`` is required, not decorative: the
    reverse-SDE integrator runs the step under ``torch.no_grad``, which would stop the
    graph being built here and silently return 0 for g(t) and f_drift.
    """
    with torch.enable_grad():
        t = t.detach().requires_grad_(True)
        out = 0.5 * torch.log(sch(t)).sum()
        if not out.requires_grad:  # schedule is constant in t (e.g. constant_tv_sch)
            return torch.zeros_like(t)
        (grad,) = torch.autograd.grad(out, t, allow_unused=True)
    return torch.zeros_like(t) if grad is None else grad.detach()


class SNRTVNoiseSchedule(BaseNoiseSchedule):
    """VP (variance-preserving) SNR/TV schedule; kernel N(a(t) x0, b(t)^2 I).

    Defaults are ISSNR(slope=2, shift=0) SNR and a constant TV=1 (so the total variance a^2 Var(x0)+b^2 is preserved at 1 for
    var-1 / normalized data). Native time domain is [t_min, t_max] in (0, 1).

    ``tv="fm"`` swaps the TV schedule for the flow-matching one (:func:`fm_tv_sch`),
    built from the *same* ``slope``/``shift`` as the SNR schedule -- the pairing is what
    makes the kernel collapse to a=(1-t)^{slope/2}, b=t^{slope/2} e^{-shift/2}. Pass
    ``snr_sch``/``tv_sch`` explicitly to override either leg (then ``slope``/``shift``
    and ``tv`` are ignored for that leg, and keeping the pair consistent is on you).
    """

    def __init__(
        self,
        snr_sch=None,
        tv_sch=None,
        slope: float = 2.0,
        shift: float = 0.0,
        tv: str = "constant",
        t_min: float = 1e-4,
        t_max: float = 0.99,
        name: str = "snr_tv_vp",
    ):
        import functools

        self.snr_sch = snr_sch or functools.partial(
            exp_inverse_sigmoid_snr_sch, slope=slope, shift=shift
        )
        if tv_sch is not None:
            self.tv_sch = tv_sch
        elif tv == "constant":
            self.tv_sch = functools.partial(constant_tv_sch, scale=1.0)
        elif tv == "fm":
            # Same (slope, shift) as the SNR leg -- that pairing is what collapses the
            # kernel to a^2=(1-t)^slope, b^2=t^slope e^{-shift}.
            self.tv_sch = functools.partial(fm_tv_sch, slope=slope, shift=shift)
        else:
            raise ValueError(f"unknown tv schedule {tv!r} (expected 'constant' or 'fm')")

        self.t_min = t_min
        self.t_max = t_max
        self.name = name

    def _clamp(self, t):
        return torch.clamp(t, min=self.t_min, max=self.t_max)

    def a2_b2(self, t):
        """Returns (a^2(t), b^2(t)) of the perturbation kernel N(a x0, b^2 I)."""
        t = self._clamp(t)
        gamma2 = self.snr_sch(t)
        tau2 = self.tv_sch(t)
        a2 = (tau2 * gamma2) / (gamma2 + 1.0)
        b2 = a2 / gamma2
        return a2, b2

    def a(self, t):
        a2, _ = self.a2_b2(t)
        return torch.sqrt(a2)

    def b(self, t):
        _, b2 = self.a2_b2(t)
        return torch.sqrt(b2)

    def g(self, t):
        """Forward-SDE diffusion g(t): g^2 = -2 tau^2 * d/dt[0.5 log gamma^2] / (1+gamma^2)."""
        t = self._clamp(t)
        gamma2 = self.snr_sch(t)
        tau2 = self.tv_sch(t)
        log_gamma_dot = _log_deriv(self.snr_sch, t)
        g2 = -(2.0 * tau2 * log_gamma_dot) / (1.0 + gamma2)
        return torch.sqrt(torch.clamp(g2, min=0.0))

    def f_drift(self, x, t):
        """Forward-SDE drift f(x,t) = x * (d/dt[0.5 log tau^2] + d/dt[0.5 log gamma^2]/(1+gamma^2))."""
        t = self._clamp(t)
        gamma2 = self.snr_sch(t)
        log_gamma_dot = _log_deriv(self.snr_sch, t)
        log_tau_dot = _log_deriv(self.tv_sch, t)
        scale = log_tau_dot + log_gamma_dot / (1.0 + gamma2)
        # scale is (B,) or scalar; broadcast against x (B, D)
        if scale.ndim >= 1 and x.ndim > scale.ndim:
            scale = scale.reshape(scale.shape + (1,) * (x.ndim - scale.ndim))
        return x * scale

    def prior_std(self, t=None):
        """Std of the prior p(x_t) = N(0, tau^2(t) I) at the reverse-SDE start (default t_max).

        Exact for unit-variance data, where a^2 Var(x0) + b^2 = tau^2. Under constant TV
        this is independent of t; under ``tv="fm"`` it is not, so pass the actual start
        time whenever the reverse SDE is truncated (see ``vp_utils.build_schedule``).
        """
        tt = torch.tensor(self.t_max if t is None else float(t))
        return float(self.tv_sch(tt) ** 0.5)

    def h(self, t):
        # VE-style h(t) is not meaningful for VP; expose b^2(t) so any accidental
        # VE-path caller gets the kernel variance rather than a crash. All VP-aware
        # code uses a2_b2 / f_drift / g directly.
        _, b2 = self.a2_b2(t)
        return b2
