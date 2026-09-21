# Control Variate Score Identity (CVSI)

Code for *Control Variate Score Matching for Diffusion Models*.

Sampling from an unnormalized density $p(x) \propto e^{-E(x)}$ with a diffusion model requires
the score $\nabla_x \log q_t(x)$, which is estimated by Monte Carlo from samples of the
denoising posterior $q(x_0 \mid x_t)$. Two identities are standard. The **Target Score
Identity** (TSI) averages energy gradients and has high variance at high noise; the
**Denoising Score Identity** (DSI) averages the posterior kernel score and has high variance at
low noise. CVSI uses one as a control variate for the other, with a time-dependent coefficient
that minimises the estimator's variance at every noise level:

$$\mathrm{CVSI} = \big(1 - \tilde c(t)\big)\,\mathrm{TSI} + \tilde c(t)\,\mathrm{DSI}.$$

$\tilde c^*(t)$ is estimated from the same samples as the score, so CVSI needs no extra energy
evaluations. TSI, DSI and heuristic mixtures are special cases.

## Contents

| | |
|---|---|
| `dem/models/components/cvsi.py` | the estimator: `idem_score_cvsi_fn` (importance-sampled, used in training) and `tilde_ct_opt_fn` / `tilde_ct_opt_weighted_fn` for $\tilde c^*(t)$ |
| `estimators.py` | TSI / DSI / CVSI from given posterior samples (`score_from_samples`, `make_estimator`) |
| `dem/models/components/noise_schedules.py` | VE schedules and the VP SNR/TV schedule of [Kahouli et al. 2025](https://arxiv.org/abs/2502.08598) |
| `vp_utils.py` | builds a VE or VP schedule with its reverse-SDE time domain and prior |
| `gmm_toy.py` | Gaussian mixture with closed-form diffused score and exact posterior sampler |
| `mnist_ebm.py`, `mnist_metrics.py` | Gaussian–Bernoulli RBM with exact block-Gibbs diffusion posterior; classifier-FID and class-TV |

The data-free sampler learning code (`dem/`, `configs/`) extends
[iDEM](https://github.com/jarridrb/DEM) ([Akhound-Sadegh et al. 2024](https://arxiv.org/abs/2402.06121)):
CVSI is an optional drop-in replacement for iDEM's TSI estimator, selected with
`model.use_cvsi=true`.

## Installation

```bash
micromamba create -f environment.yaml
micromamba activate dem
pip install -r requirements.txt
```

Requires PyTorch ≥ 2.0. Training logs to Weights & Biases by default: copy `.env.example` to
`.env` and set `WANDB_ENTITY`, or pass `logger=csv` to log locally. Run all commands from the
repository root.

## Experiments

### Training-free sampling

The score is computed directly from the energy and the posterior at each step of the reverse
SDE; no network is trained.

| notebook | |
|---|---|
| `gmm_training_free_sampling.ipynb` | GMM with random means $\mu_i \sim \mathcal N(0, s^2 d\,I)$ and Wishart covariances: estimator error against the analytic score, and reverse-SDE generation |
| `gmm_estimator_scaling.ipynb` | the same comparison with dimension and number of components as parameters |
| `mnist_gbrbm_sampling.ipynb` | Gaussian–Bernoulli RBM on 14×14 MNIST ($D=196$): estimator variance, reverse-SDE generation, classifier-FID and class-TV against the Monte Carlo budget |

Each notebook has a `SCHEDULE = "VE" | "VP"` flag. The GMM notebooks default to high-dimensional
mixtures (dimension and number of components are set in the configuration cell) and are meant for
a GPU; set `TOY = True` and `SMOKE = True` for a small configuration that runs on a CPU. The
MNIST notebook trains the RBM and the classifier on first run and caches them in `models/`.

### Data-free sampler learning

```bash
# 40-component 2D GMM
python dem/train.py experiment=gmm_idem model.use_cvsi=false   # iDEM (TSI)
python dem/train.py experiment=gmm_idem model.use_cvsi=true    # iDEM + CVSI

# vary the number of energy evaluations per training sample
python dem/train.py experiment=gmm_idem model.use_cvsi=true model.num_estimator_mc_samples=8

# DW-4
python dem/train.py experiment=dw4_idem model.use_cvsi=true

# 6D GMM with a reduced effective diffusion scale (lambda_eff = 0.5)
python dem/train.py experiment=gmm_norm_idem energy.dimensionality=6 \
    model.use_cvsi=true model.diffusion_scale=0.5
```

`model.diffusion_scale` is $\lambda_\mathrm{eff}$: it scales the noise term of the reverse SDE
independently of the drift. Evaluate a checkpoint with
`python dem/eval.py experiment=gmm_idem ckpt_path=<path>`. The DW-4 data splits are in `data/`.

CVSI options (in `configs/model/dem.yaml`, override as `model.<name>=<value>`):

| option | default | |
|---|---|---|
| `use_cvsi` | `false` | CVSI instead of TSI |
| `num_estimator_mc_samples` | `100` | posterior samples per score estimate |
| `cvsi_weighted_stats` | `false` | use the importance weights when estimating $\tilde c^*(t)$ |
| `ct_per_molecule` | `true` | estimate $\tilde c^*(t)$ per particle, then average (particle systems) |
| `clip_all` | `false` | also norm-clip the combined CVSI score |

## License

MIT. See `LICENSE`, which also retains the license of the iDEM code this repository extends.
