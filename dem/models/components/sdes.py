import torch


class SDE(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, drift, diffusion):
        super().__init__()
        self.drift = drift
        self.diffusion = diffusion

    def f(self, t, x):
        if t.dim() == 0:
            # repeat the same time for all points if we have a scalar time
            t = t * torch.ones(x.shape[0]).to(x.device)

        return self.drift(t, x)

    def g(self, t, x):
        return self.diffusion(t, x)


class VEReverseSDE(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, score, noise_schedule):
        super().__init__()
        self.score = score
        self.noise_schedule = noise_schedule

    def f(self, t, x):
        if t.dim() == 0:
            # repeat the same time for all points if we have a scalar time
            t = t * torch.ones(x.shape[0]).to(x.device)

        score = self.score(t, x)
        return self.g(t, x).pow(2) * score

    def g(self, t, x):
        g = self.noise_schedule.g(t)
        return g.unsqueeze(1) if g.ndim > 0 else torch.full_like(x, g)


class ReverseSDE(VEReverseSDE):
    """Reverse SDE for a general (VE or VP) schedule.

    Reverse drift (lambda=1, matching iDEM/VEReverseSDE):
        f_rev(x,t) = g(t)^2 * score(x,t) - f_fwd(x,t)
    where f_fwd is the forward-SDE drift. For a VE schedule f_fwd = 0 and this is
    identical to VEReverseSDE (g^2 * score). For a VP schedule that exposes
    ``f_drift`` the forward contraction is subtracted (it becomes an expansion in
    reverse time). ``g(t)`` is taken from the schedule as in VEReverseSDE.
    """

    def f(self, t, x):
        if t.dim() == 0:
            t = t * torch.ones(x.shape[0]).to(x.device)
        drift = self.g(t, x).pow(2) * self.score(t, x)
        if hasattr(self.noise_schedule, "f_drift"):
            drift = drift - self.noise_schedule.f_drift(x, t)
        return drift


class RegVEReverseSDE(VEReverseSDE):
    def f(self, t, x):
        dx = super().f(t, x[..., :-1])
        quad_reg = 0.5 * dx.pow(2).sum(dim=-1, keepdim=True)
        return torch.cat([dx, quad_reg], dim=-1)

    def g(self, t, x):
        g = self.noise_schedule.g(t)
        if g.ndim > 0:
            return g.unsqueeze(1)
        return torch.cat(
            [torch.full_like(x[..., :-1], g), torch.zeros_like(x[..., -1:])], dim=-1
        )
