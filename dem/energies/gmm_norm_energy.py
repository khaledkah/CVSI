import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as dist
from fab.utils.plotting import plot_contours, plot_marginal_pair
from lightning.pytorch.loggers import WandbLogger

from dem.energies.base_energy_function import BaseEnergyFunction
from dem.utils.logging_utils import fig_to_image


class GMMNorm(BaseEnergyFunction):
    def __init__(
        self,
        dimensionality=2,
        n_mixes=40,
        device="cpu",
        plotting_buffer_sample_size=512,
        plot_samples_epoch_period=5,
        train_set_size=100000,
        test_set_size=2000,
        val_set_size=2000,
        data_path_train=None,
        seed=0,
        mean_scale=10.0,
        df_scale=2.0,
        data_normalization_factor=None,
        should_unnormalize=True,
    ):
        torch.manual_seed(seed)
        self.n_mixes = n_mixes
        self.device = device
        self.plotting_buffer_sample_size = plotting_buffer_sample_size
        self.plot_samples_epoch_period = plot_samples_epoch_period
        self.train_set_size = train_set_size
        self.test_set_size = test_set_size
        self.val_set_size = val_set_size
        self.data_path_train = data_path_train
        self.mean_scale = mean_scale
        self.df_scale = df_scale
        self.data_normalization_factor = data_normalization_factor
        self.should_unnormalize = should_unnormalize
        self.name = "gmm_norm"

        # Generate random GMM parameters
        # 1. Generate weights via uniform draw then normalize to sum to 1
        weights = torch.rand(n_mixes)
        weights = weights / weights.sum()

        # 2. Generate means scaled by sqrt(d) to control separation
        scaled_mean = self.mean_scale * (dimensionality ** 0.5)
        means = torch.randn(n_mixes, dimensionality) * scaled_mean

        # 3. Generate covariance matrices using Wishart distribution
        df = max(dimensionality, int(dimensionality * self.df_scale))
        # Sample Wishart(df, I) via sum of outer products of standard normals
        covs = []
        for _ in range(n_mixes):
            x = torch.randn(df, dimensionality)
            cov = x.T @ x
            covs.append(cov)
        covs = torch.stack(covs)

        # Normalize GMM parameters (Logic from csvi.ipynb)
        # gmm_mean = (weights[:,None]*means).sum() / means.shape[-1]
        # Note: In notebook, means is (C, D), weights is (C,).
        # weights[:, None] * means -> (C, D). sum() -> scalar.
        # This assumes we want to center the "average coordinate" to 0.

        gmm_mean = (weights.unsqueeze(1) * means).sum() / dimensionality

        # cov_components = jnp.einsum("...i,...j->...ij",
        #                             (means - gmm_mean), (means - gmm_mean))
        means_centered = means - gmm_mean
        # Outer product: (C, D, 1) * (C, 1, D) -> (C, D, D)
        cov_components = torch.matmul(
            means_centered.unsqueeze(2), means_centered.unsqueeze(1)
        )

        # gmm_cov = (weights[:,None,None]* (covs + cov_components)).sum(0)
        gmm_cov = (weights.view(-1, 1, 1) * (covs + cov_components)).sum(0)

        # gmm_std = jnp.sqrt(jnp.trace(gmm_cov) / gmm_cov.shape[-1])
        gmm_std = torch.sqrt(torch.trace(gmm_cov) / dimensionality)

        # Update parameters
        self.means = (means - gmm_mean) / gmm_std
        self.covs = covs / (gmm_std**2)
        self.weights = weights

        # Move to device
        self.weights = self.weights.to(device)
        self.means = self.means.to(device)
        self.covs = self.covs.to(device)

        # Create distribution
        self.mix = dist.Categorical(self.weights)
        self.comp = dist.MultivariateNormal(self.means, self.covs)
        self.gmm = dist.MixtureSameFamily(self.mix, self.comp)

        self.curr_epoch = 0

        super().__init__(
            dimensionality=dimensionality,
            normalization_min=-data_normalization_factor,
            normalization_max=data_normalization_factor,
        )

    def setup_test_set(self):
        return self.gmm.sample((self.test_set_size,))

    def setup_train_set(self):
        if self.data_path_train is None:
            train_samples = self.normalize(self.gmm.sample((self.train_set_size,)))
        else:
            if self.data_path_train.endswith(".pt"):
                data = torch.load(self.data_path_train)
                if isinstance(data, np.ndarray):
                    data = torch.from_numpy(data)
            else:
                data = np.load(self.data_path_train, allow_pickle=True)
                data = torch.tensor(data)

            train_samples = data.to(self.device)

        return train_samples

    def setup_val_set(self):
        return self.gmm.sample((self.val_set_size,))

    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        if self.should_unnormalize:
            samples = self.unnormalize(samples)
        return self.gmm.log_prob(samples)

    @property
    def dimensionality(self):
        return self._dimensionality

    def log_on_epoch_end(
        self,
        latest_samples: torch.Tensor,
        latest_energies: torch.Tensor,
        wandb_logger: WandbLogger,
        unprioritized_buffer_samples=None,
        cfm_samples=None,
        replay_buffer=None,
        prefix: str = "",
    ) -> None:
        if wandb_logger is None:
            return

        if len(prefix) > 0 and prefix[-1] != "/":
            prefix += "/"

        if self.curr_epoch % self.plot_samples_epoch_period == 0:
            if self.should_unnormalize:
                # Don't unnormalize CFM samples since they're in the
                # unnormalized space
                if latest_samples is not None:
                    latest_samples = self.unnormalize(latest_samples)

                if unprioritized_buffer_samples is not None:
                    unprioritized_buffer_samples = self.unnormalize(
                        unprioritized_buffer_samples
                    )

            if unprioritized_buffer_samples is not None:
                buffer_samples, _, _ = replay_buffer.sample(
                    self.plotting_buffer_sample_size
                )

                samples_fig = self.get_dataset_fig(buffer_samples, latest_samples)
                wandb_logger.log_image(
                    f"{prefix}unprioritized_buffer_samples", [samples_fig]
                )

            if cfm_samples is not None:
                cfm_samples_fig = self.get_dataset_fig(
                    unprioritized_buffer_samples, cfm_samples
                )
                wandb_logger.log_image(
                    f"{prefix}cfm_generated_samples", [cfm_samples_fig]
                )

            if latest_samples is not None:
                # Scatter plot of first 2 dims
                fig, ax = plt.subplots()
                ax.scatter(
                    latest_samples.detach().cpu()[:, 0],
                    latest_samples.detach().cpu()[:, 1],
                )
                wandb_logger.log_image(
                    f"{prefix}generated_samples_scatter", [fig_to_image(fig)]
                )

                img = self.get_single_dataset_fig(
                    latest_samples, "dem_generated_samples"
                )
                wandb_logger.log_image(f"{prefix}generated_samples", [img])

            plt.close()

        self.curr_epoch += 1

    def log_samples(
        self,
        samples: torch.Tensor,
        wandb_logger: WandbLogger,
        name: str = "",
        should_unnormalize: bool = False,
    ) -> None:
        if wandb_logger is None:
            return
        if self.should_unnormalize and should_unnormalize:
            samples = self.unnormalize(samples)
        samples_fig = self.get_single_dataset_fig(samples, name)
        wandb_logger.log_image(f"{name}", [samples_fig])

    def _get_marginal_log_prob_fn(self):
        # Returns a function that computes log prob of the marginal distribution
        # of the first 2 dimensions
        if self.dimensionality == 2:

            def log_prob_fn(x):
                x = x.to(self.device)
                return self.gmm.log_prob(x).cpu()

            return log_prob_fn
        else:
            # Construct marginal GMM for plotting
            marginal_means = self.means[:, :2]
            marginal_covs = self.covs[:, :2, :2]

            # We need to be careful with marginal_covs.
            # The marginal covariance of a multivariate normal is just the submatrix.

            marginal_comp = dist.MultivariateNormal(marginal_means, marginal_covs)
            marginal_gmm = dist.MixtureSameFamily(self.mix, marginal_comp)

            def log_prob_fn(x):
                # x is (N, 2)
                x = x.to(self.device)
                return marginal_gmm.log_prob(x).cpu()

            return log_prob_fn

    def get_single_dataset_fig(self, samples, name, plotting_bounds=None):
        # If dim != 2, we project to first 2 dims for visualization
        samples_2d = samples[:, :2]

        if plotting_bounds is None:
            # Since data is unit variance, bounds [-3, 3] should be fine
            plotting_bounds = (-3, 3)

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        log_prob_fn = self._get_marginal_log_prob_fn()

        plot_contours(
            log_prob_fn,
            bounds=plotting_bounds,
            ax=ax,
            n_contour_levels=20,
            grid_width_n_points=200,
        )

        plot_marginal_pair(samples_2d, ax=ax, bounds=plotting_bounds)
        ax.set_title(f"{name}")

        return fig_to_image(fig)

    def get_dataset_fig(self, samples, gen_samples=None, plotting_bounds=None):
        samples_2d = samples[:, :2]
        if gen_samples is not None:
            gen_samples_2d = gen_samples[:, :2]
        else:
            gen_samples_2d = None

        if plotting_bounds is None:
            plotting_bounds = (-3, 3)

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        log_prob_fn = self._get_marginal_log_prob_fn()

        plot_contours(
            log_prob_fn,
            bounds=plotting_bounds,
            ax=axs[0],
            n_contour_levels=20,
            grid_width_n_points=200,
        )

        # plot dataset samples
        plot_marginal_pair(samples_2d, ax=axs[0], bounds=plotting_bounds)
        axs[0].set_title("Buffer")

        if gen_samples_2d is not None:
            plot_contours(
                log_prob_fn,
                bounds=plotting_bounds,
                ax=axs[1],
                n_contour_levels=20,
                grid_width_n_points=200,
            )
            # plot generated samples
            plot_marginal_pair(gen_samples_2d, ax=axs[1], bounds=plotting_bounds)
            axs[1].set_title("Generated samples")

        # delete subplot
        else:
            fig.delaxes(axs[1])

        return fig_to_image(fig)
