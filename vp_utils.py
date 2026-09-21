from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from dem.models.components.noise_schedules import (
    GeometricNoiseSchedule,
    SNRTVNoiseSchedule,
)


# ---------------------------------------------------------------------------
# Schedule + reverse-SDE domain
# ---------------------------------------------------------------------------
@dataclass
class ScheduleSpec:
    name: str
    noise: object
    prior_std: float   # std of the reverse-SDE initial cloud, in *scaled* coordinates
    gen_start: float   # reverse SDE start time
    gen_end: float     # reverse SDE end time
    scale: float       # physical = scale * scaled  (1.0 under VE)

    def summary(self) -> str:
        a2_lo, b2_lo = _ab(self.noise, self.gen_end)
        a2_hi, b2_hi = _ab(self.noise, self.gen_start)
        return (
            f"{self.name}: t {self.gen_start:.4g} -> {self.gen_end:.4g} | "
            f"prior_std={self.prior_std:.3f} | coord scale={self.scale:.3f} | "
            f"(a2,b2) start=({a2_hi:.3g},{b2_hi:.3g}) end=({a2_lo:.3g},{b2_lo:.3g})"
        )


def _ab(noise, t):
    from dem.models.components.cvsi import get_a2_b2

    a2, b2 = get_a2_b2(torch.as_tensor(float(t)).reshape(1), noise)
    return float(a2), float(b2)


def build_schedule(
    name: str,
    sigma_min: float = 0.001,
    sigma_max: float = 6.0,
    scale: float = 1.0,
    gen_start: Optional[float] = None,
    tv: str = "constant",
) -> ScheduleSpec:
    """VE: geometric schedule on ``[1, 0]``, prior ``N(0, h(1) I)``, no rescaling.

    VP: SNR/TV schedule on ``[t_max, t_min]``, prior ``N(0, tau^2 I)`` with tau=1, and
    ``scale`` (the data std) mapping the unit-variance coordinates the
    schedule assumes back to physical ones.

    ``gen_start`` truncates where the reverse SDE starts (default: the schedule's own
    high-noise end). Under VP this matters: as ``t -> t_max`` the scaling ``a(t) -> 0``, so
    the posterior tilts towards ``N(x_t/a, (b/a)^2)``, which degenerates, and integrating
    through that sliver injects large outward drift. Constant TV holds the marginal at ``tau^2`` for all
    t, so truncating to ~0.9 costs almost nothing. Under VE there is no such degeneracy.

    ``tv`` (VP only) picks the total-variance schedule: ``"constant"`` (TV=1, default) or
    ``"fm"`` (flow matching, kernel ``a=1-t``, ``b=t`` at slope 2 / shift 0). ``"fm"`` does
    not preserve the total variance, hence ``prior_std(start)`` below.
    """
    if name == "VE":
        noise = GeometricNoiseSchedule(sigma_min=sigma_min, sigma_max=sigma_max)
        prior_std = float(noise.h(torch.tensor(1.0)).sqrt())
        start = 1.0 if gen_start is None else float(gen_start)
        # The prior must match the marginal at the *start* time, which under VE grows
        # with t -- so a truncated start needs sqrt(h(start)), not sqrt(h(1)).
        if gen_start is not None:
            prior_std = float(noise.h(torch.tensor(start)).sqrt())
        return ScheduleSpec("VE", noise, prior_std, start, 0.0, 1.0)
    if name == "VP":
        noise = SNRTVNoiseSchedule(tv=tv)
        start = noise.t_max if gen_start is None else float(gen_start)
        # Constant TV: the marginal std is tau(t) at every t, so truncating the start
        # leaves the prior unchanged. FM TV: tau varies, hence prior_std(start).
        return ScheduleSpec(
            "VP", noise, noise.prior_std(start), start, noise.t_min, float(scale)
        )
    raise ValueError(f"unknown schedule {name!r} (expected 'VE' or 'VP')")
