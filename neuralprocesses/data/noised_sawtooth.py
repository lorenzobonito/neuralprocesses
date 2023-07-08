import lab as B
import torch

from neuralprocesses.aggregate import Aggregate, AggregateInput

from .data import SyntheticGenerator, new_multi_batch
from ..dist import UniformContinuous

__all__ = ["NoisedSawtoothGenerator"]


class NoisedSawtoothGenerator(SyntheticGenerator):

    def __init__(self, *args, dist_freq=UniformContinuous(3, 5), noise_levels=None, beta=None, **kw_args):
        super().__init__(*args, **kw_args)
        self.dist_freq = dist_freq
        self.noise_levels = noise_levels
        self.beta = beta

    def _noise_up(self, yt, iters):

        for _ in range(iters):
            yt = B.sqrt(1-self.beta)*yt+ self.beta*torch.randn(yt.shape).to(self.device)

        return yt

    def generate_batch(self):
        with B.on_device(self.device):

            xc, nc, multi_xt = new_multi_batch(self, self.dim_y, self.noise_levels+1)
            x = B.concat(xc, multi_xt[0], axis=1)
            _c = lambda x: B.cast(self.dtype, x)

            # Sample a frequency.
            self.state, freq = self.dist_freq.sample(
                self.state,
                self.float64,
                self.batch_size,
                self.dim_y_latent,
            )

            # Sample a direction.
            self.state, direction = B.randn(
                self.state,
                self.float64,
                self.batch_size,
                self.dim_y_latent,
                B.shape(x, 2),
            )
            norm = B.sqrt(B.sum(direction * direction, axis=2, squeeze=False))
            direction = direction / norm

            # Sample a uniformly distributed (conditional on frequency) offset.
            self.state, sample = B.rand(
                self.state,
                self.float64,
                self.batch_size,
                self.dim_y_latent,
                1,
            )
            offset = sample / freq

            multi_yt = []
            for level, xt in enumerate(multi_xt):

                x = B.concat(xc, xt, axis=1)

                # Construct the sawtooth and add noise.
                f = (freq * (B.matmul(direction, x, tr_b=True) - offset)) % 1
                if self.h is not None:
                    f = B.matmul(self.h, f)
                y = f + B.sqrt(self.noise) * B.randn(f)

                yc = _c(y[:, :, :nc])
                yt = self._noise_up(_c(y[:, :, nc:]), level)

                if level == 0:
                    context = [(_c(B.transpose(xc)), yc)]
                else:
                    context.extend([(_c(B.transpose(xt)), yt)])
                multi_yt.append(yt)

            # Create batch.
            batch = {}
            batch["contexts"] = context
            batch["xt"] = AggregateInput(*((_c(B.transpose(xt)), i) for i, xt in enumerate(multi_xt)))
            batch["yt"] = Aggregate(*multi_yt)

            return batch
