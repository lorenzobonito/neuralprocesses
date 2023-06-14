import lab as B
from neuralprocesses.aggregate import Aggregate, AggregateInput
import neuralprocesses.torch as nps
import numpy as np
import torch
import matplotlib.pyplot as plt

from neuralprocesses.dist.normal import MultiOutputNormal
from wbml.plot import tweak
from context_utils import mask_contexts

__all__ = ["generate_AR_prediction"]


def generate_AR_prediction(state, model, batch, num_samples, normalise=True, path=None, config=None):

    true_y0t = batch["yt"][0]
    float = B.dtype_float(true_y0t)
    float64 = B.promote_dtypes(float, np.float64)
    
    if config:
        try:
            plot_config = config["plot"][1]
        except KeyError:
            return
    
    with B.on_device(batch["xt"]):
        x = B.linspace(B.dtype(batch["xt"]), *plot_config["range"], 200)
        x = x[None, None, :]
        x = AggregateInput(*((x, i) for i in range(3)))

    with torch.no_grad():

        logpdfs = None
        for i in range(num_samples):
            
            # Generating predictions for y2t
            contexts = mask_contexts(batch["contexts"], 1, 2)
            state, mean, var, _, _ = nps.predict(state,
                                                  model,
                                                  contexts,
                                                  x,
                                                  num_samples=1,
                                                  batch_size=1)
            state, _, _, _, yt = nps.predict(state,
                                             model,
                                             contexts,
                                             batch["xt"],
                                             num_samples=1,
                                             batch_size=1)
            y2t_pred = yt[2].squeeze(0).float()

            if config:
                plt.figure(figsize=(8, 6 * 3))
                plt.subplot(3, 1, 3)
                plt.scatter(contexts[0][0], contexts[0][1], label="Original Context", style="train", c="blue", s=20)
                if contexts[1][0].numel() != 0:
                    plt.scatter(contexts[1][0], contexts[1][1], label="Auxiliary Context 1", style="train", marker="^", c="tab:red", s=10)
                if contexts[2][0].numel() != 0:
                    plt.scatter(contexts[2][0], contexts[2][1], label="Auxiliary Context 2", style="train", marker="^", c="tab:green", s=10)
                plt.scatter(batch["xt"][2][0], batch["yt"][2], label="Target", style="test", s=20)
                err = 1.96 * B.sqrt(var[2][0, 0])
                plt.plot(x[2][0], mean[2][0, 0], label="Prediction", style="pred")
                plt.fill_between(x[2][0], mean[2][0, 0] - err, mean[2][0, 0] + err, style="pred")
                plt.scatter(batch["xt"][2][0], y2t_pred, marker="s", c="black", s=10, label="Prediction sample")

                for x_axvline in plot_config["axvline"]:
                    plt.axvline(x_axvline, c="k", ls="--", lw=0.5)
                plt.xlim(B.min(x[2][0]), B.max(x[2][0]))
                tweak()

            # Generating predictions for y1t
            contexts[2] = (batch["xt"][2][0], y2t_pred)
            state, mean, var, _, _ = nps.predict(state,
                                                  model,
                                                  contexts,
                                                  x,
                                                  num_samples=1,
                                                  batch_size=1)
            state, _, _, _, yt = nps.predict(state,
                                             model,
                                             contexts,
                                             batch["xt"],
                                             num_samples=1,
                                             batch_size=1)
            y1t_pred = yt[1].squeeze(0).float()
        
            if config:
                plt.subplot(3, 1, 2)
                plt.scatter(contexts[0][0], contexts[0][1], label="Original Context", style="train", c="blue", s=20)
                if contexts[1][0].numel() != 0:
                    plt.scatter(contexts[1][0], contexts[1][1], label="Auxiliary Context 1", style="train", marker="^", c="tab:red", s=10)
                if contexts[2][0].numel() != 0:
                    plt.scatter(contexts[2][0], contexts[2][1], label="Auxiliary Context 2", style="train", marker="^", c="tab:green", s=10)
                plt.scatter(batch["xt"][1][0], batch["yt"][1], label="Target", style="test", s=20)
                err = 1.96 * B.sqrt(var[1][0, 0])
                plt.plot(x[1][0], mean[1][0, 0], label="Prediction", style="pred")
                plt.fill_between(x[1][0], mean[1][0, 0] - err, mean[1][0, 0] + err, style="pred")
                plt.scatter(batch["xt"][1][0], y1t_pred, marker="s", c="black", s=10, label="Prediction sample")

                for x_axvline in plot_config["axvline"]:
                    plt.axvline(x_axvline, c="k", ls="--", lw=0.5)
                plt.xlim(B.min(x[1][0]), B.max(x[1][0]))
                tweak()

            # Generating predictions for y0t
            contexts[1] = (batch["xt"][1][0], y1t_pred)
            state, pred = model(state, contexts, x)

            if config:
                plt.subplot(3, 1, 1)
                plt.scatter(contexts[0][0], contexts[0][1], label="Original Context", style="train", c="blue", s=20)
                if contexts[1][0].numel() != 0:
                    plt.scatter(contexts[1][0], contexts[1][1], label="Auxiliary Context 1", style="train", marker="^", c="tab:red", s=10)
                if contexts[2][0].numel() != 0:
                    plt.scatter(contexts[2][0], contexts[2][1], label="Auxiliary Context 2", style="train", marker="^", c="tab:green", s=10)
                plt.scatter(batch["xt"][0][0], batch["yt"][0], label="Target", style="test", s=20)
                err = 1.96 * B.sqrt(var[0][0, 0])
                plt.plot(x[0][0], mean[0][0, 0], label="Prediction", style="pred")
                plt.fill_between(x[0][0], mean[0][0, 0] - err, mean[0][0, 0] + err, style="pred")

                for x_axvline in plot_config["axvline"]:
                    plt.axvline(x_axvline, c="k", ls="--", lw=0.5)
                plt.xlim(B.min(x[0][0]), B.max(x[0][0]))
                tweak()

                plt.savefig(path)
                plt.close()
            
            if i == 0:
                # Disable plot after first sample
                config=False
            
            state, pred = model(state, contexts, batch["xt"])

            # Select level of interest and isolate relevant predictions
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
