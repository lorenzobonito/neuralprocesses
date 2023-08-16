import lab as B
import stheno
import torch

from neuralprocesses.aggregate import Aggregate, AggregateInput

from .data import SyntheticGenerator, new_multi_batch

__all__ = ["NoisedGPGenerator"]


class NoisedGPGenerator(SyntheticGenerator):

    def __init__(
        self,
        *args,
        kernel=stheno.EQ().stretch(0.25),
        pred_logpdf=True,
        pred_logpdf_diag=True,
        noise_levels=None,
        beta=None,
        same_xt = False,
        **kw_args,
    ):
        super().__init__(*args, **kw_args)
        self.kernel = kernel
        self.pred_logpdf = pred_logpdf
        self.pred_logpdf_diag = pred_logpdf_diag
        self.noise_levels = noise_levels
        self.beta = beta
        self.same_xt = same_xt

    def _noise_up(self, yt, iters):

        for _ in range(iters):
            yt = B.sqrt(1-self.beta)*yt+ self.beta*torch.randn(yt.shape).to(self.device)

        return yt

    def generate_batch(self):

        with B.on_device(self.device):

            xc, _, multi_xt = new_multi_batch(self, self.dim_y, self.noise_levels+1)
            if self.same_xt:
                for idx in range(1, len(multi_xt)):
                    multi_xt[idx] = multi_xt[0]
            _c = lambda x: B.cast(self.dtype, x)

            with stheno.Measure() as prior:
                f = stheno.GP(self.kernel)
                # Construct FDDs for the context and target points.
                fc = f(xc, self.noise)
                ft = f(multi_xt[0], self.noise)

            multi_yt = []
            self.state, yc, yt = prior.sample(self.state, fc, ft)
            context = [(_c(B.transpose(xc)), _c(B.transpose(yc)))]
            multi_yt.append(_c(B.transpose(yt)))

            for level in range(1, len(multi_xt)):
                yt_n = self._noise_up(_c(multi_yt[0]), level)
                context.extend([(_c(B.transpose(multi_xt[0])), yt_n)])
                multi_yt.append(yt_n)

            # Create batch.
            batch = {}
            batch["contexts"] = context
            batch["xt"] = AggregateInput(*((_c(B.transpose(xt)), i) for i, xt in enumerate(multi_xt)))
            batch["yt"] = Aggregate(*multi_yt)

            # Compute predictive logpdfs.
            if self.pred_logpdf or self.pred_logpdf_diag:
                post = prior | (fc, yc)
            if self.pred_logpdf:
                batch["pred_logpdf"] = post(ft).logpdf(yt)
            if self.pred_logpdf_diag:
                batch["pred_logpdf_diag"] = post(ft).diagonalise().logpdf(yt)

            return batch
