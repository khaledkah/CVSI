from __future__ import annotations

import math
import os
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _as_2d(x: torch.Tensor) -> torch.Tensor:
    """Flatten everything but the batch dim -> (N, F) float tensor."""
    x = torch.as_tensor(x).float()
    return x.reshape(x.shape[0], -1)


def _sqdist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise squared Euclidean distances (numerically clamped >= 0)."""
    return torch.cdist(a, b).clamp_min(0).pow(2)


def _knn_radius(feats: torch.Tensor, k: int) -> torch.Tensor:
    """Distance from each point to its k-th nearest neighbour (excluding self)."""
    d = torch.cdist(feats, feats)
    d.fill_diagonal_(float("inf"))
    k = min(k, feats.shape[0] - 1)
    return d.kthvalue(k, dim=1).values  # (N,)


# --------------------------------------------------------------------------- #
# two-sample distances (MMD / KID / Frechet)
# --------------------------------------------------------------------------- #
def mmd2_rbf(
    gen: torch.Tensor,
    ref: torch.Tensor,
    bandwidths: Optional[Sequence[float]] = None,
) -> float:
    """Unbiased squared MMD with a (multi-bandwidth) RBF kernel.

    0 iff the two samples are indistinguishable under the kernel; larger = more
    different. Bandwidths default to {0.5, 1, 2} x the median squared pairwise
    distance of the pooled sample (median heuristic).
    """
    X, Y = _as_2d(gen), _as_2d(ref).to(_as_2d(gen).device)
    m, n = X.shape[0], Y.shape[0]
    dxx, dyy, dxy = _sqdist(X, X), _sqdist(Y, Y), _sqdist(X, Y)

    if bandwidths is None:
        med = torch.median(torch.cat([dxx.flatten(), dyy.flatten(), dxy.flatten()]))
        med = med.clamp_min(1e-8)
        bandwidths = [med * s for s in (0.5, 1.0, 2.0)]

    total = 0.0
    for h in bandwidths:
        Kxx = torch.exp(-dxx / (2 * h))
        Kyy = torch.exp(-dyy / (2 * h))
        Kxy = torch.exp(-dxy / (2 * h))
        Kxx.fill_diagonal_(0.0)
        Kyy.fill_diagonal_(0.0)
        total += (
            Kxx.sum() / (m * (m - 1))
            + Kyy.sum() / (n * (n - 1))
            - 2.0 * Kxy.mean()
        )
    return (total / len(bandwidths)).item()


def kid(
    gen: torch.Tensor,
    ref: torch.Tensor,
    subset_size: int = 1000,
    n_subsets: int = 100,
    degree: int = 3,
    seed: int = 0,
) -> Dict[str, float]:
    """Kernel Inception Distance with a polynomial kernel (Inception-free).

    Unbiased MMD^2 with kernel k(x,y) = (x.y / d + 1)^degree, averaged over
    random subsets -> robust for small sample sizes. Returns mean and std.
    """
    X, Y = _as_2d(gen), _as_2d(ref).to(_as_2d(gen).device)
    d = X.shape[1]
    g = torch.Generator(device="cpu").manual_seed(seed)
    size = min(subset_size, X.shape[0], Y.shape[0])

    def poly(a, b):
        return (a @ b.t() / d + 1.0).pow(degree)

    ests = []
    for _ in range(n_subsets):
        xi = torch.randperm(X.shape[0], generator=g)[:size]
        yi = torch.randperm(Y.shape[0], generator=g)[:size]
        x, y = X[xi], Y[yi]
        Kxx, Kyy, Kxy = poly(x, x), poly(y, y), poly(x, y)
        Kxx.fill_diagonal_(0.0)
        Kyy.fill_diagonal_(0.0)
        mmd = (
            Kxx.sum() / (size * (size - 1))
            + Kyy.sum() / (size * (size - 1))
            - 2.0 * Kxy.mean()
        )
        ests.append(mmd.item())
    ests = torch.tensor(ests)
    return {"kid_mean": ests.mean().item(), "kid_std": ests.std().item()}


def _sym_sqrt(mat: torch.Tensor) -> torch.Tensor:
    """Matrix square root of a symmetric PSD matrix via eigendecomposition."""
    vals, vecs = torch.linalg.eigh(mat)
    vals = vals.clamp_min(0).sqrt()
    return (vecs * vals) @ vecs.t()


def frechet_distance(gen: torch.Tensor, ref: torch.Tensor) -> float:
    """Frechet distance between two Gaussians fit to the feature vectors.

    This is the FID formula; feed it classifier features (not Inception) for a
    domain-appropriate "classifier-FID".
    """
    X, Y = _as_2d(gen).double(), _as_2d(ref).double().to(_as_2d(gen).device)
    mu1, mu2 = X.mean(0), Y.mean(0)
    C1 = torch.cov(X.t())
    C2 = torch.cov(Y.t())
    diff = mu1 - mu2
    covmean = _sym_sqrt(_sym_sqrt(C1) @ C2 @ _sym_sqrt(C1))
    fid = diff.dot(diff) + torch.trace(C1 + C2 - 2.0 * covmean)
    return fid.clamp_min(0).item()


# --------------------------------------------------------------------------- #
# manifold metrics: fidelity vs diversity
# --------------------------------------------------------------------------- #
def precision_recall(gen: torch.Tensor, ref: torch.Tensor, k: int = 3) -> Dict[str, float]:
    """Improved precision & recall (Kynkaanniemi et al. 2019).

    precision = fraction of generated points inside the reference manifold
                (each ref point spans a ball of radius = its k-th NN distance).
    recall    = fraction of reference points inside the generated manifold.
    High precision / low recall => sharp but mode-collapsed; the reverse =>
    diverse but off-manifold/blurry.
    """
    G, R = _as_2d(gen), _as_2d(ref).to(_as_2d(gen).device)
    r_ref = _knn_radius(R, k)
    r_gen = _knn_radius(G, k)
    d = torch.cdist(G, R)  # (Ng, Nr)
    precision = (d <= r_ref[None, :]).any(dim=1).float().mean().item()
    recall = (d.t() <= r_gen[None, :]).any(dim=1).float().mean().item()
    return {"precision": precision, "recall": recall}


def density_coverage(gen: torch.Tensor, ref: torch.Tensor, k: int = 5) -> Dict[str, float]:
    """Density & coverage (Naeem et al. 2020) -- robust fidelity / diversity.

    density  = (1/k) * average number of reference balls each generated point
               falls into (unbounded; ~1.0 is ideal, can exceed 1).
    coverage = fraction of reference points that have at least one generated
               point inside their k-NN ball (in [0, 1]).
    """
    G, R = _as_2d(gen), _as_2d(ref).to(_as_2d(gen).device)
    r_ref = _knn_radius(R, k)
    d = torch.cdist(G, R)  # (Ng, Nr)
    within = d <= r_ref[None, :]
    density = (within.float().sum() / (k * G.shape[0])).item()
    coverage = within.any(dim=0).float().mean().item()
    return {"density": density, "coverage": coverage}


def knn_novelty(gen: torch.Tensor, ref: torch.Tensor, copy_quantile: float = 0.01) -> Dict[str, float]:
    """Nearest-neighbour distance from each generated sample to the reference.

    Guards against memorisation: a generated set that merely copies the
    reference has near-zero NN distances. ``frac_copies`` is the fraction whose
    NN distance is below the ``copy_quantile`` of reference-internal NN
    distances (i.e. closer to a reference point than reference points typically
    are to each other).
    """
    G, R = _as_2d(gen), _as_2d(ref).to(_as_2d(gen).device)
    nn_gen = torch.cdist(G, R).min(dim=1).values
    ref_internal = _knn_radius(R, 1)
    thresh = torch.quantile(ref_internal, copy_quantile)
    return {
        "nn_dist_mean": nn_gen.mean().item(),
        "nn_dist_min": nn_gen.min().item(),
        "frac_copies": (nn_gen < thresh).float().mean().item(),
    }


# --------------------------------------------------------------------------- #
# mode coverage via classifier labels
# --------------------------------------------------------------------------- #
def _hist(labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    h = torch.bincount(labels.long().flatten().cpu(), minlength=n_classes).float()
    return h / h.sum().clamp_min(1.0)


def class_distribution_stats(
    labels_gen: torch.Tensor,
    labels_ref: torch.Tensor,
    n_classes: int = 10,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Compare predicted-class histograms of generated vs reference samples.

    Catches mode *imbalance*, not just presence/absence: an estimator emitting
    90% ones still "covers" all 10 classes but has a large KL here.
    """
    p = _hist(labels_gen, n_classes) + eps
    q = _hist(labels_ref, n_classes) + eps
    p, q = p / p.sum(), q / q.sum()
    kl = (p * (p / q).log()).sum().item()
    tv = 0.5 * (p - q).abs().sum().item()
    m = 0.5 * (p + q)
    js = 0.5 * (p * (p / m).log()).sum().item() + 0.5 * (q * (q / m).log()).sum().item()
    ent = -(p * p.log()).sum().item()
    return {
        "class_kl": kl,
        "class_tv": tv,
        "class_js": js,
        "n_modes": int((_hist(labels_gen, n_classes) > 0).sum().item()),
        "entropy": ent,
        "entropy_uniform": math.log(n_classes),
    }


# --------------------------------------------------------------------------- #
# small MNIST classifier: domain-appropriate feature extractor + labeller
# --------------------------------------------------------------------------- #
class MNISTClassifier(nn.Module):
    """Tiny CNN replacing Inception. Gives labels (``predict``) and a penultimate
    feature vector (``features``). Accepts flat (N, H*W) or image (N, 1, H, W)
    inputs; ``adaptive_avg_pool`` makes it resolution-agnostic."""

    def __init__(self, side: int, n_classes: int = 10, feat_dim: int = 64):
        super().__init__()
        self.side = side
        self.feat_dim = feat_dim
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.fc_feat = nn.Linear(32 * 4 * 4, feat_dim)
        self.fc_out = nn.Linear(feat_dim, n_classes)

    def _to_img(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.reshape(-1, 1, self.side, self.side)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        return x

    def features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(self._to_img(x)).flatten(1)
        return F.relu(self.fc_feat(h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_out(self.features(x))

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).argmax(dim=1)


def train_classifier(
    images: torch.Tensor,
    labels: torch.Tensor,
    side: int,
    n_classes: int = 10,
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 0,
    verbose: bool = True,
) -> MNISTClassifier:
    """Train the small CNN on real images (in [0, 1])."""
    torch.manual_seed(seed)
    model = MNISTClassifier(side, n_classes).to(device)
    X = _as_2d(images).to(device)
    y = labels.long().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = X.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        tot, correct, loss_sum = 0, 0, 0.0
        model.train()
        for i in range(0, n, batch_size):
            b = perm[i : i + batch_size]
            opt.zero_grad()
            logits = model(X[b])
            loss = F.cross_entropy(logits, y[b])
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(b)
            correct += (logits.argmax(1) == y[b]).sum().item()
            tot += len(b)
        if verbose:
            print(f"  classifier epoch {ep + 1}/{epochs}  "
                  f"loss={loss_sum / tot:.4f}  acc={correct / tot:.4f}")
    model.eval()
    return model


def load_or_train_classifier(
    path: str,
    images: torch.Tensor,
    labels: torch.Tensor,
    side: int,
    n_classes: int = 10,
    device: str = "cpu",
    **train_kwargs,
) -> MNISTClassifier:
    """Load a cached classifier from ``path`` or train and cache one."""
    model = MNISTClassifier(side, n_classes).to(device)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, weights_only=True, map_location=device))
        model.eval()
        print(f"loaded cached classifier from {path}")
        return model
    model = train_classifier(images, labels, side, n_classes, device=device, **train_kwargs)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"saved classifier to {path}")
    return model


# --------------------------------------------------------------------------- #
# top-level driver
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_samples(
    gen_imgs: torch.Tensor,
    ref_imgs: torch.Tensor,
    classifier: Optional[MNISTClassifier] = None,
    n_classes: int = 10,
    k_pr: int = 3,
    k_dc: int = 5,
) -> Dict[str, float]:
    """Run every metric on one (generated, reference) image pair.

    ``gen_imgs`` / ``ref_imgs`` are image tensors in [0, 1], flat (N, D) or
    (N, H, W). Pixel-space MMD/KID are always computed; if a ``classifier`` is
    given, the manifold/label metrics run in its feature space (the meaningful
    space) instead of raw pixels.
    """
    gen, ref = _as_2d(gen_imgs), _as_2d(ref_imgs)
    out: Dict[str, float] = {}

    # pixel-space two-sample distances (cheap, no classifier needed)
    out["pixel_mmd2"] = mmd2_rbf(gen, ref)
    out.update({f"pixel_{k}": v for k, v in kid(gen, ref).items()})

    if classifier is None:
        # manifold metrics in raw-pixel space as a fallback
        out.update(precision_recall(gen, ref, k=k_pr))
        out.update(density_coverage(gen, ref, k=k_dc))
        out.update(knn_novelty(gen, ref))
        return out

    fg = classifier.features(gen)
    fr = classifier.features(ref)
    out["feat_fid"] = frechet_distance(fg, fr)
    out["feat_mmd2"] = mmd2_rbf(fg, fr)
    out.update(precision_recall(fg, fr, k=k_pr))
    out.update(density_coverage(fg, fr, k=k_dc))
    out.update(knn_novelty(fg, fr))
    out.update(class_distribution_stats(
        classifier.predict(gen), classifier.predict(ref), n_classes=n_classes))
    return out


def format_table(results: Dict[str, Dict[str, float]], keys: Optional[Sequence[str]] = None) -> str:
    """Pretty-print ``{name: metric_dict}`` as an aligned table (rows = metrics)."""
    names = list(results)
    if keys is None:
        keys = list(results[names[0]])
    width = max(len(k) for k in keys) + 2
    header = f"{'metric':<{width}}" + "".join(f"{n:>14s}" for n in names)
    lines = [header, "-" * len(header)]
    for k in keys:
        row = f"{k:<{width}}"
        for n in names:
            v = results[n].get(k, float("nan"))
            row += f"{v:>14.4g}" if isinstance(v, (int, float)) else f"{str(v):>14s}"
        lines.append(row)
    return "\n".join(lines)
