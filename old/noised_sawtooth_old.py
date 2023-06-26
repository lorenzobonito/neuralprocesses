import lab as B
import torch

from neuralprocesses.aggregate import Aggregate, AggregateInput

from ..neuralprocesses.data.data import SyntheticGenerator, new_batch
from ..neuralprocesses.dist import UniformContinuous

__all__ = ["NoisedSawtoothGenerator"]


class NoisedSawtoothGenerator(SyntheticGenerator):

    def __init__(self, *args, dist_freq=UniformContinuous(3, 5), noise_levels=2, **kw_args):
        super().__init__(*args, **kw_args)
        self.dist_freq = dist_freq
        self.noise_levels = noise_levels

    def _noise_up(self, yt, beta=0.1):

        return B.sqrt(1-beta)*yt + beta*torch.randn(yt.shape).to(self.device)

    def generate_batch(self):
        with B.on_device(self.device):
            set_batch, xcs, xc, nc, xts, xt, nt = new_batch(self, self.dim_y)
            x = B.concat(xc, xt, axis=1)

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

            # Construct the sawtooth and add noise.
            f = (freq * (B.matmul(direction, x, tr_b=True) - offset)) % 1
            if self.h is not None:
                f = B.matmul(self.h, f)
            y = f + B.sqrt(self.noise) * B.randn(f)

            y_c = y[:, :, :nc]
            y0_t = y[:, :, nc:]

            # Create batch.
            batch = {}
            set_batch(batch, y_c, y0_t, transpose=False)

            xt = batch["xt"]
            yts = [y0_t]

            # Create noised targets
            for _ in range(self.noise_levels):
                yts.append(self._noise_up(yts[-1]))

            _c = lambda x: B.cast(self.dtype, x)
            
            # Add further context sets
            batch["contexts"].extend([(_c(xt), _c(yt)) for yt in yts[1:]])

            # Add further targets
            batch["xt"] = AggregateInput(*((_c(xt), i) for i in range(self.noise_levels + 1)))
            batch["yt"] = Aggregate(*(_c(yt) for yt in yts))
            
            return batch
