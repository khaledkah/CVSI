from __future__ import annotations

import numpy as np
import torch

_LOG2PI = float(np.log(2 * np.pi))


class GMMEnergy:
    """p(x) = sum_i w_i N(mu_i, Sigma_i); is_molecule=False. Normalized at construction.

    ``isotropic`` controls how "round" the generated mixture is:

    * ``False`` (default) -- anisotropic; only the scalar whitening of ``normalize``.
    * ``True`` -- whiten the **mixture** so its total covariance is exactly ``I``. This is
      the sense VP needs (constant TV preserves ``a^2 Var(x0)+b^2`` per *direction*), and
      it supersedes ``normalize``.
    * ``"components"`` -- make each component spherical but leave the mean scatter alone.
      Does *not* make the mixture isotropic.
    """

    is_molecule = False
    n_particles = None
    n_spatial_dim = None

    def __init__(self, means, covs, weights, normalize=True, isotropic=False):
        w = weights / weights.sum()
        if isotropic == "components":
            # Spherical components only: Sigma_i -> (tr Sigma_i / D) I. This alone does
            # NOT make the mixture isotropic -- the per-dim variance spread is dominated
            # by the scatter of the means. Use isotropic=True for the VP-relevant sense.
            D = means.shape[-1]
            var = torch.diagonal(covs, dim1=-2, dim2=-1).sum(-1) / D      # (M,)
            covs = var[:, None, None] * torch.eye(
                D, dtype=covs.dtype, device=covs.device
            )
        elif isotropic:
            # Whiten the MIXTURE so its total covariance is exactly I (mean-scatter term
            # included). This is the sense VP needs: constant TV preserves
            # a^2 Var(x0) + b^2 = 1 per direction, which a single scalar only achieves
            # when the marginal has no preferred direction.
            mix_mean = (w[:, None] * means).sum(0)
            diff = means - mix_mean
            mix_cov = (
                w[:, None, None] * (covs + torch.einsum("md,me->mde", diff, diff))
            ).sum(0)
            # W = mix_cov^{-1/2} (symmetric inverse square root)
            evals, evecs = torch.linalg.eigh(mix_cov)
            W = evecs @ torch.diag(evals.clamp_min(1e-12).rsqrt()) @ evecs.T
            means = (means - mix_mean) @ W
            covs = W @ covs @ W
            normalize = False  # already exactly unit-variance; scalar pass would undo it
        if normalize:
            # scalar (isotropic) whitening to mean~0, per-dim var~1 (reference-style)
            mix_mean = (w[:, None] * means).sum(0)  # (D,)
            diff = means - mix_mean
            mix_cov = (
                w[:, None, None] * (covs + torch.einsum("md,me->mde", diff, diff))
            ).sum(0)
            mix_std = torch.sqrt(torch.trace(mix_cov) / means.shape[-1])
            means = (means - mix_mean) / mix_std
            covs = covs / mix_std**2
        self.means = means
        self.covs = covs
        self.w = w
        self.M, self.D = means.shape
        self.Prec = torch.linalg.inv(covs)
        self.logdet = torch.logdet(covs)
        self.L = torch.linalg.cholesky(covs)

    def _eye(self, ref):
        return torch.eye(self.D, dtype=ref.dtype, device=ref.device)

    def _log_comp(self, x):
        d = x.unsqueeze(-2) - self.means
        maha = torch.einsum("...md,mde,...me->...m", d, self.Prec, d)
        return (
            torch.log(self.w)
            - 0.5 * self.logdet
            - 0.5 * self.D * _LOG2PI
            - 0.5 * maha
        )

    def __call__(self, x):
        return torch.logsumexp(self._log_comp(x), dim=-1)

    def sample(self, n):
        comp = torch.multinomial(self.w, n, replacement=True)
        z = torch.randn(n, self.D, dtype=self.means.dtype, device=self.means.device)
        return self.means[comp] + torch.einsum("nde,ne->nd", self.L[comp], z)

    def nearest_mode(self, x):
        return (x[:, None, :] - self.means[None]).norm(dim=-1).argmin(-1)

    # ---- analytic diffusion helpers for kernel N(sqrt(a2) x0, b2 I); a2,b2 scalars ----
    def _mlr(self, xt, a2, b2):  # log responsibilities of marginal q_t
        am = torch.sqrt(a2)
        Mi = a2 * self.covs + b2 * self._eye(xt)
        d = xt.unsqueeze(-2) - am * self.means
        maha = torch.einsum("bmd,mde,bme->bm", d, torch.linalg.inv(Mi), d)
        return (
            torch.log(self.w)
            - 0.5 * torch.logdet(Mi)
            - 0.5 * self.D * _LOG2PI
            - 0.5 * maha
        )

    def marginal_score(self, xt, a2, b2):  # s*(xt) = grad log q_t
        am = torch.sqrt(a2)
        Mi = a2 * self.covs + b2 * self._eye(xt)
        Pi = torch.linalg.inv(Mi)
        d = xt.unsqueeze(-2) - am * self.means
        r = torch.softmax(self._mlr(xt, a2, b2), dim=-1)
        comp_score = -torch.einsum("mde,bme->bmd", Pi, d)
        return torch.einsum("bm,bmd->bd", r, comp_score)

    def sample_posterior_exact(self, xt, a2, b2, K):
        am = torch.sqrt(a2)
        r = torch.softmax(self._mlr(xt, a2, b2), dim=-1)  # (B,M)
        Ci = torch.linalg.inv(self.Prec + (a2 / b2) * self._eye(xt))  # (M,D,D)
        rhs = (
            torch.einsum("mde,me->md", self.Prec, self.means)[None]
            + am * xt[:, None] / b2
        )
        mi = torch.einsum("mde,bme->bmd", Ci, rhs)  # (B,M,D)
        B = xt.shape[0]
        Lc = torch.linalg.cholesky(Ci)
        comp = torch.multinomial(r, K, replacement=True)  # (B,K)
        z = torch.randn(K, B, self.D, dtype=xt.dtype, device=xt.device)
        m_sel = torch.gather(
            mi, 1, comp.unsqueeze(-1).expand(-1, -1, self.D)
        ).permute(1, 0, 2)
        # Group samples per mixture component instead of `Lc[comp.T]`, which would
        # gather a full (D,D) matrix per sample and materialize a (K,B,D,D) tensor.
        # M is small, so grouping keeps peak memory at O(K*B*D).
        comp_flat = comp.T.reshape(-1)  # (K*B,)
        z_flat = z.reshape(-1, self.D)
        Lz_flat = torch.empty_like(z_flat)
        for m in range(self.M):
            mask = comp_flat == m
            if mask.any():
                Lz_flat[mask] = z_flat[mask] @ Lc[m].T
        return m_sel + Lz_flat.reshape(K, B, self.D)
