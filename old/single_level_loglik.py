import lab as B
import numpy as np
from neuralprocesses.aggregate import Aggregate, AggregateInput

from neuralprocesses.dist.normal import MultiOutputNormal

from neuralprocesses.model import Model
# from .util import fix_noise as fix_noise_in_pred
from neuralprocesses import _dispatch
from neuralprocesses.numdata import num_data

__all__ = ["sl_loglik"]


@_dispatch
def sl_loglik(
    state: B.RandomState,
    model: Model,
    contexts: list,
    xt,
    yt,
    level_index,
    *,
    num_samples=1,
    batch_size=16,
    normalise=False,
    # fix_noise=None,
    **kw_args,
):
    """Log-likelihood objective.

    Args:
        state (random state, optional): Random state.
        model (:class:`.Model`): Model.
        xc (input): Inputs of the context set.
        yc (tensor): Output of the context set.
        xt (input): Inputs of the target set.
        yt (tensor): Outputs of the target set.
        num_samples (int, optional): Number of samples. Defaults to 1.
        batch_size (int, optional): Batch size to use for sampling. Defaults to 16.
        normalise (bool, optional): Normalise the objective by the number of targets.
            Defaults to `False`.
        fix_noise (float, optional): Fix the likelihood variance to this value.

    Returns:
        random state, optional: Random state.
        tensor: Log-likelihoods.
    """
    float = B.dtype_float(yt)
    float64 = B.promote_dtypes(float, np.float64)

    # Sample in batches to alleviate memory requirements.
    logpdfs = None
    done_num_samples = 0
    while done_num_samples < num_samples:
        # Limit the number of samples at the batch size.
        this_num_samples = min(num_samples - done_num_samples, batch_size)

        # Perform batch.
        state, pred = model(
            state,
            contexts,
            xt,
            num_samples=this_num_samples,
            dtype_enc_sample=float,
            dtype_lik=float64,
            **kw_args,
        )
        # pred = fix_noise_in_pred(pred, fix_noise)

        # Select level of interest and isolate relevant predictions
        mean = list(pred.mean)[level_index].squeeze(2)
        var = list(pred.var)[level_index].squeeze(2)
        shape = list(pred.shape)[level_index]
        SL_pred = MultiOutputNormal.diagonal(mean, var, shape)
        this_logpdfs = SL_pred.logpdf(B.cast(float64, yt[level_index]))

        # If the number of samples is equal to one but `num_samples > 1`, then the
        # encoding was a `Dirac`, so we can stop batching. Also, set `num_samples = 1`
        # because we only have one sample now. We also don't need to do the
        # `logsumexp` anymore.
        if num_samples > 1 and B.shape(this_logpdfs, 0) == 1:
            logpdfs = this_logpdfs
            num_samples = 1
            break

        # Record current samples.
        if logpdfs is None:
            logpdfs = this_logpdfs
        else:
            # Concatenate at the sample dimension.
            logpdfs = B.concat(logpdfs, this_logpdfs, axis=0)

        # Increase the counter.
        done_num_samples += this_num_samples

    # Average over samples. Sample dimension should always be the first.
    logpdfs = B.logsumexp(logpdfs, axis=0) - B.log(num_samples)

    if normalise:
        # Normalise by the number of targets.
        # NOTE: The code below is not OK if the number of targets varies between the three layers,
        # and would need to be changed accordingly (e.g. keep track of how many targets alongside logpdfs)
        logpdfs = logpdfs / B.cast(float64, num_data(AggregateInput(xt[0]), Aggregate(yt[0])))

    return state, logpdfs


@_dispatch
def sl_loglik(state: B.RandomState, model: Model, xc, yc, xt, yt, **kw_args):
    return sl_loglik(state, model, [(xc, yc)], xt, yt, **kw_args)


@_dispatch
def sl_loglik(model: Model, *args, **kw_args):
    state = B.global_random_state(B.dtype(args[-2]))
    state, logpdfs = sl_loglik(state, model, *args, **kw_args)
    B.set_global_random_state(state)
    return logpdfs
