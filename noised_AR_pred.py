import lab as B
from neuralprocesses.aggregate import Aggregate, AggregateInput
import neuralprocesses.torch as nps
import numpy as np
import torch
import matplotlib.pyplot as plt

from neuralprocesses.dist.normal import MultiOutputNormal
from wbml.plot import tweak
from context_utils import mask_contexts

__all__ = ["generate_AR_prediction", "split_AR_prediction"]


def split_AR_prediction(state, models, batch, num_samples, normalise=True, path=None, config=None):

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

    with torch.no_grad():

        logpdfs = None
        for i in range(num_samples):
            
            # Generating predictions for y2t
            contexts = mask_contexts(batch["contexts"], 1, 2)
            state, mean, var, _, _ = nps.predict(state,
                                                  models[2],
                                                  contexts,
                                                  x,
                                                  num_samples=1,
                                                  batch_size=1)
            state, _, _, _, yt = nps.predict(state,
                                             models[2],
                                             contexts,
                                             batch["xt"][2][0],
                                             num_samples=1,
                                             batch_size=1)
            y2t_pred = yt.squeeze(0).float()

            if config:
                plt.figure(figsize=(8, 6 * 3))
                plt.subplot(3, 1, 3)
                plt.scatter(contexts[0][0], contexts[0][1], label="Context", style="train", s=20)
                for c in range(1, 3):
                    plt.scatter(contexts[c][0], contexts[c][1], label="Noised context", style="train", marker="^", s=20)
                plt.scatter(batch["xt"][2][0], batch["yt"][2], label="Target", style="test", s=20)
                err = 1.96 * B.sqrt(var[0, 0])
                plt.plot(x, mean[0, 0], label="Prediction", style="pred")
                plt.fill_between(x, mean[0, 0] - err, mean[0, 0] + err, style="pred")
                plt.scatter(batch["xt"][2][0], y2t_pred, marker="s", c="tab:red", s=20, label="Prediction sample")

                for x_axvline in plot_config["axvline"]:
                    plt.axvline(x_axvline, c="k", ls="--", lw=0.5)
                plt.xlim(B.min(x), B.max(x))
                tweak()

            # Generating predictions for y1t
            
            # TRIED TURNING OFF AND JUST USING OG CONTEXT IN LAYER 1 TOO (remember to add these back in below)
            # contexts[2] = (batch["xt"][2][0], y2t_pred)
            state, mean, var, _, _ = nps.predict(state,
                                                  models[1],
                                                  contexts,
                                                  x,
                                                  num_samples=1,
                                                  batch_size=1)
            state, _, _, _, yt = nps.predict(state,
                                             models[1],
                                             contexts,
                                             batch["xt"][1][0],
                                             num_samples=1,
                                             batch_size=1)
            y1t_pred = yt.squeeze(0).float()
        
            if config:
                plt.subplot(3, 1, 2)
                plt.scatter(contexts[0][0], contexts[0][1], label="Context", style="train", s=20)
                for c in range(1, 3):
                    plt.scatter(contexts[c][0], contexts[c][1], label="Noised context", style="train", marker="^", s=20)
                plt.scatter(batch["xt"][1][0], batch["yt"][1], label="Target", style="test", s=20)
                err = 1.96 * B.sqrt(var[0, 0])
                plt.plot(x, mean[0, 0], label="Prediction", style="pred")
                plt.fill_between(x, mean[0, 0] - err, mean[0, 0] + err, style="pred")
                plt.scatter(batch["xt"][1][0], y1t_pred, marker="s", c="tab:green", s=20, label="Prediction sample")

                for x_axvline in plot_config["axvline"]:
                    plt.axvline(x_axvline, c="k", ls="--", lw=0.5)
                plt.xlim(B.min(x), B.max(x))
                tweak()

            # Generating predictions for y0t
            contexts[1] = (batch["xt"][1][0], y1t_pred)

            # Added to test layer 1 only OG context
            contexts[2] = (batch["xt"][2][0], y2t_pred)
            state, pred = models[0](state, contexts, x)

            if config:
                plt.subplot(3, 1, 1)
                plt.scatter(contexts[0][0], contexts[0][1], label="Context", style="train", s=20)
                for c in range(1, 3):
                    plt.scatter(contexts[c][0], contexts[c][1], label="Noised context", style="train", marker="^", s=20)
                plt.scatter(batch["xt"][0][0], batch["yt"][0], label="Target", style="test", s=20)
                err = 1.96 * B.sqrt(pred.var[0, 0])
                plt.plot(x, pred.mean[0, 0], label="Prediction", style="pred")
                plt.fill_between(x, pred.mean[0, 0] - err, pred.mean[0, 0] + err, style="pred")

                for x_axvline in plot_config["axvline"]:
                    plt.axvline(x_axvline, c="k", ls="--", lw=0.5)
                plt.xlim(B.min(x), B.max(x))
                tweak()

                plt.savefig(path)
                plt.close()
            
            if i == 0:
                # Disable plot after first sample
                config=False
            
            state, pred = models[0](state, contexts, batch["xt"][0][0])

            this_logpdfs = pred.logpdf(B.cast(float64, true_y0t))
            # this_logpdfs = pred.logpdf(B.cast(float64, true_y0t).float())
            # print(this_logpdfs)
            # import sys
            # sys.exit(1)

            if logpdfs is None:
                logpdfs = this_logpdfs
            else:
                logpdfs = B.concat(logpdfs, this_logpdfs, axis=0)
        
        # print(logpdfs)

        # Average over samples.
        logpdfs = B.logsumexp(logpdfs, axis=0) - B.log(num_samples)
        # print(logpdfs)

        if normalise:
            # Normalise by the number of targets.
            logpdfs = logpdfs / B.cast(float64, nps.num_data(AggregateInput(batch["xt"][0]), Aggregate(batch["yt"][0])))
        
        # print(logpdfs)
        # import sys
        # sys.exit()

    return state, logpdfs


def generate_AR_prediction(state, model, batch, num_samples, normalise=True, path=None, config=None):

    true_y0t = list(batch["yt"])[0]
    float = B.dtype_float(true_y0t)
    float64 = B.promote_dtypes(float, np.float64)
    
    if config:
        try:
            plot_config = config["plot"][1]
            plt.figure(figsize=(8, 6 * config["dim_y"]))
        except KeyError:
            return
    
    with B.on_device(batch["xt"]):
        x = B.linspace(B.dtype(batch["xt"]), *plot_config["range"], 100)

    with torch.no_grad():

        logpdfs = None
        for i in range(num_samples):

            # Generating predictions for y2t
            contexts = mask_context(batch["contexts"], 1, 2)
            state, mean, var, _, yt = nps.predict(state,
                                                   model,
                                                   contexts,
                                                   batch["xt"],
                                                   num_samples=1,
                                                   batch_size=1)
            y2t_pred = list(yt)[2].squeeze(0).float()

            if config:
                xcontexts = batch["contexts"][0][0].squeeze(0)
                ycontexts = batch["contexts"][0][1].squeeze(0)
                plt.subplot(3, 1, 3)
                plt.scatter(xcontexts, ycontexts, label="Context", style="train", s=20)
                plt.scatter(batch["xt"][2][0], batch["yt"][2], label="Target", style="test", s=20)
                err = 1.96 * B.sqrt(var[2][0, 0])
                plt.plot(x, mean[2][0, 0], label="Prediction", style="pred")
                plt.fill_between(x, mean[2][0, 0] - err, mean[2][0, 0] + err, style="pred")
                plt.scatter(batch["xt"][2][0], y2t_pred, marker="s", c="tab:red", s=20, label="Prediction sample")

                for x_axvline in plot_config["axvline"]:
                    plt.axvline(x_axvline, c="k", ls="--", lw=0.5)
                plt.xlim(B.min(x), B.max(x))
                tweak()

            # Generating predictions for y1t
            contexts[2] = (list(batch["xt"])[0][0], y2t_pred)
            state, mean, var, _, yt = nps.predict(state,
                                                   model,
                                                   contexts,
                                                   batch["xt"],
                                                   num_samples=1,
                                                   batch_size=1)
            y1t_pred = list(yt)[1].squeeze(0).float()
        
            if config:
                xcontexts = torch.cat((batch["contexts"][0][0].squeeze(0), batch["contexts"][2][0].squeeze(0)), 1)
                ycontexts = torch.cat((batch["contexts"][0][1].squeeze(0), batch["contexts"][2][1].squeeze(0)), 1)
                plt.subplot(3, 1, 2)
                plt.scatter(xcontexts, ycontexts, label="Context", style="train", s=20)
                plt.scatter(batch["xt"][1][0], batch["yt"][1], label="Target", style="test", s=20)
                err = 1.96 * B.sqrt(var[1][0, 0])
                plt.plot(x, mean[1][0, 0], label="Prediction", style="pred")
                plt.fill_between(x, mean[1][0, 0] - err, mean[1][0, 0] + err, style="pred")
                plt.scatter(batch["xt"][1][0], y1t_pred, marker="s", c="tab:green", s=20, label="Prediction sample")

                for x_axvline in plot_config["axvline"]:
                    plt.axvline(x_axvline, c="k", ls="--", lw=0.5)
                plt.xlim(B.min(x), B.max(x))
                tweak()

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
            
            if config:
                xcontexts = torch.cat((batch["contexts"][0][0].squeeze(0), batch["contexts"][1][0].squeeze(0), batch["contexts"][2][0].squeeze(0)), 1)
                ycontexts = torch.cat((batch["contexts"][0][1].squeeze(0), batch["contexts"][1][1].squeeze(0), batch["contexts"][2][1].squeeze(0)), 1)
                plt.subplot(3, 1, 1)
                plt.scatter(xcontexts, ycontexts, label="Context", style="train", s=20)
                plt.scatter(batch["xt"][0][0], batch["yt"][0], label="Target", style="test", s=20)
                err = 1.96 * B.sqrt(pred.var[0][0, 0])
                plt.plot(x, pred.mean[0][0, 0], label="Prediction", style="pred")
                plt.fill_between(x, pred.mean[0][0, 0] - err, pred.mean[0][0, 0] + err, style="pred")
            
                for x_axvline in plot_config["axvline"]:
                    plt.axvline(x_axvline, c="k", ls="--", lw=0.5)
                plt.xlim(B.min(x), B.max(x))
                tweak()

                plt.savefig(path)
            
            if i == 0:
                # Disable plot after first sample
                config=False

        # Average over samples.
        logpdfs = B.logsumexp(logpdfs, axis=0) - B.log(num_samples)

        if normalise:
            # Normalise by the number of targets.
            logpdfs = logpdfs / B.cast(float64, nps.num_data(AggregateInput(batch["xt"][0]), Aggregate(batch["yt"][0])))

    return state, logpdfs
