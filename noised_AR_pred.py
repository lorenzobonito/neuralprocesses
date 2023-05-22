import lab as B
from neuralprocesses.aggregate import Aggregate, AggregateInput
import neuralprocesses.torch as nps
import numpy as np
import torch

from neuralprocesses.dist.normal import MultiOutputNormal
from mask_context import mask_context

__all__ = ["generate_AR_prediction"]


def generate_AR_prediction(state, model, batch, num_samples, normalise=True):

    true_y0t = list(batch["yt"])[0]
    float = B.dtype_float(true_y0t)
    float64 = B.promote_dtypes(float, np.float64)
    print(batch["contexts"][0][0].numel())

    with torch.no_grad():

        logpdfs = None
        for _ in range(num_samples):

            # Generating predictions for y2t
            contexts = mask_context(batch["contexts"], 1, 2)
            state, _, _, _, yt = nps.predict(state,
                                             model,
                                             contexts,
                                             batch["xt"],
                                             num_samples=1,
                                             batch_size=1)
            y2t_pred = list(yt)[2].squeeze(0).float()

            # Generating predictions for y1t
            contexts[2] = (list(batch["xt"])[0][0], y2t_pred)
            state, _, _, _, yt = nps.predict(state,
                                             model,
                                             contexts,
                                             batch["xt"],
                                             num_samples=1,
                                             batch_size=1)
            y1t_pred = list(yt)[1].squeeze(0).float()

            # Generating predictions for y0t
            contexts[1] = (list(batch["xt"])[0][0], y1t_pred)
            # state, _, _, _, yt = nps.predict(state,
            #                                  model,
            #                                  contexts,
            #                                  batch["xt"],
            #                                  num_samples=1,
            #                                  batch_size=1)
            # y0t_pred = list(yt)[0].squeeze(0)
            state, pred = model(
                state,
                contexts,
                batch["xt"],
            )
            mean = list(pred.mean)[0].squeeze(2)
            var = list(pred.var)[0].squeeze(2)
            shape = list(pred.shape)[0]
            SL_pred = MultiOutputNormal.diagonal(mean, var, shape)
            this_logpdfs = SL_pred.logpdf(B.cast(float64, true_y0t))

            if logpdfs is None:
                logpdfs = this_logpdfs
            else:
                logpdfs = B.concat(logpdfs, this_logpdfs, axis=0)

        # Average over samples.
        logpdfs = B.logsumexp(logpdfs, axis=0) - B.log(num_samples)

        if normalise:
            # Normalise by the number of targets.
            logpdfs = logpdfs / B.cast(float64, nps.num_data(AggregateInput(batch["xt"][0]), Aggregate(batch["yt"][0])))

    return state, logpdfs
