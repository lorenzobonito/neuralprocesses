import lab as B
from neuralprocesses.aggregate import Aggregate, AggregateInput
import neuralprocesses.torch as nps
import numpy as np
import torch
import matplotlib.pyplot as plt

from wbml.plot import tweak
from batch_masking import mask_contexts, mask_xt, mask_yt

__all__ = ["generate_AR_prediction"]


def generate_AR_prediction(state, model, batch, num_samples, normalise=True, path=None, config=None):

    true_y0t = mask_yt(batch["yt"], 0)
    num_layers = len(batch["xt"])
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
        x = AggregateInput(*((x, i) for i in range(num_layers)))

    with torch.no_grad():

        logpdfs = None
        for _ in range(num_samples):

            # Mask context for noisiest layer
            contexts = mask_contexts(batch["contexts"], num_layers-1)

            if config:
                plt.figure(figsize=(8, 6 * num_layers))
                colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][2:]

            for level_index in range(num_layers-1, -1, -1):

                if config:
                    l_x = mask_xt(x, level_index)
                    state, pred = model(state, contexts, l_x)

                if level_index != 0:
                    l_xt = mask_xt(batch["xt"], level_index)
                    state, _, _, _, yt = nps.predict(state,
                                                    model,
                                                    contexts,
                                                    l_xt,
                                                    num_samples=1,
                                                    batch_size=1)
                    l_yt = yt[level_index].squeeze(0).float()
                    contexts[level_index] = (l_xt[level_index][0], l_yt)

                if config:
                    plt.subplot(num_layers, 1, level_index+1)
                    plt.scatter(contexts[0][0].squeeze(0), contexts[0][1].squeeze(0), label="Original Context", style="train", c="blue", s=20)
                    for j in range(num_layers):
                        if j>level_index:
                            plt.scatter(contexts[j][0].squeeze(0), contexts[j][1].squeeze(0), label=f"Auxiliary Context {j}", style="train", marker="^", c=colors[j-1], s=10)

                    plt.scatter(
                        batch["xt"][level_index][0],
                        batch["yt"][level_index],
                        label="Target",
                        style="test",
                        s=20,
                    )

                    # Plot prediction.
                    err = 1.96 * B.sqrt(pred.var[level_index][0, 0])
                    plt.plot(
                        l_x[level_index][0],
                        pred.mean[level_index][0, 0],
                        label="Prediction",
                        style="pred",
                    )
                    plt.fill_between(
                        l_x[level_index][0],
                        pred.mean[level_index][0, 0] - err,
                        pred.mean[level_index][0, 0] + err,
                        style="pred",
                    )

                    if level_index != 0:
                        plt.scatter(l_xt[level_index][0], l_yt, marker="s", c="black", s=10, label="Prediction sample")

                    for x_axvline in plot_config["axvline"]:
                        plt.axvline(x_axvline, c="k", ls="--", lw=0.5)

                    plt.xlim(B.min(l_x[level_index][0]), B.max(l_x[level_index][0]))
                    tweak()

            if config:
                plt.savefig(path)
                plt.close()

            config = False
            
            l_xt = mask_xt(batch["xt"], 0)
            state, pred = model(state, contexts, l_xt)

            this_logpdfs = pred.logpdf(B.cast(float64, true_y0t))

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
