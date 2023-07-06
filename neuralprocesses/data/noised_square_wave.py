import lab as B
import numpy as np
import torch

from neuralprocesses.aggregate import Aggregate, AggregateInput

from .data import SyntheticGenerator, new_multi_batch
from ..dist import UniformContinuous

__all__ = ["NoisedSquareWaveGenerator"]


class NoisedSquareWaveGenerator(SyntheticGenerator):

    def __init__(self, *args, dist_freq=UniformContinuous(2, 4), dist_off=UniformContinuous(-1, 1), dist_ampl=UniformContinuous(0.5, 2), noise_levels=2, **kw_args):
        super().__init__(*args, **kw_args)
        self.dist_freq = dist_freq
        self.dist_off = dist_off
        self.dist_ampl = dist_ampl
        self.noise_levels = noise_levels

    def _noise_up(self, yt, iters, beta=0.1):

        for _ in range(iters):
            yt = B.sqrt(1-beta)*yt+ beta*torch.randn(yt.shape).to(self.device)

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
            # Sample an offset.
            self.state, offset = self.dist_off.sample(
                self.state,
                self.float64,
                self.batch_size,
                self.dim_y_latent,
            )
            # Sample an amplitude.
            self.state, amplitude = self.dist_ampl.sample(
                self.state,
                self.float64,
                self.batch_size,
                self.dim_y_latent,
            )

            multi_yt = []
            for level, xt in enumerate(multi_xt):

                x = B.concat(xc, xt, axis=1)

                # Construct the sawtooth and add noise.
                f = B.transpose(B.where(B.floor(x * freq) % 2 == 0, offset, offset+amplitude))
                # if self.h is not None:
                #     f = B.matmul(self.h, f)
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
