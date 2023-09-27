import lab as B
from neuralprocesses.aggregate import Aggregate, AggregateInput
import matplotlib.cm as cm
import neuralprocesses.torch as nps
import numpy as np
import torch
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

from wbml.plot import tweak
from batch_masking import mask_contexts, mask_xt, mask_yt

__all__ = ["joint_AR_prediction", "split_AR_prediction"]


def joint_AR_prediction(state, model, batch, num_samples, ar_context, prop_context: bool = False, normalise=True, path=None, config=None):

    if prop_context and ar_context != 0:
        raise ValueError("AR context process not implemented for proportional context.")

    true_y0t = mask_yt(batch["yt"], 0)
    num_layers = len(batch["xt"])
    batch_size = batch["contexts"][0][0].shape[0]
    empty = B.randn(torch.float32, batch_size, 1, 0)
    float = B.dtype_float(true_y0t)
    float64 = B.promote_dtypes(float, np.float64)
    
    if config:
        try:
            plot_config = config["plot"][1]
            plot_config["range"] = (-2, 2)
        except KeyError:
            return
    
    with B.on_device(batch["xt"]):
        x = B.linspace(B.dtype(batch["xt"]), *plot_config["range"], 200)
        x = x[None, None, :]
        x = AggregateInput(*((x, i) for i in range(num_layers)))

        # If we want to predict only at original target locations, the line below should be left uncommented.
        # Both options should be investigated.
        batch["xt"] = AggregateInput(*((batch["xt"][0][0], i) for i in range(num_layers)))

    og_context_size = B.length(batch["contexts"][0][0])

    with torch.no_grad():

        logpdfs = None

        if og_context_size < ar_context:
            xt_reshape = batch["xt"][num_layers-1][0].squeeze((0, 1))
            perm = torch.IntTensor(B.randperm(len(xt_reshape))).to(xt_reshape.device)
            xt_choice = torch.index_select(xt_reshape, 0, perm[:ar_context-og_context_size])
            # Line below allows for duplicates, which is not ideal
            # state, xt_choice = B.choice(state, batch["xt"][num_layers-1][0].squeeze((0, 1)), (ar_context-og_context_size))
            xt_subsample = [(empty, i) for i in range(num_layers-1)]
            xt_subsample.append((B.expand_dims(xt_choice, axis=0, times=2), num_layers-1))
            xt_subsample = AggregateInput(*xt_subsample)
            state, _, _, ft, _ = nps.ar_predict(state, model, batch["contexts"], xt_subsample, num_samples=1, order="random")
            expaned_contexts = [(B.expand_dims(B.concat(*(batch["contexts"][0][0].squeeze((0, 1)), xt_subsample[num_layers-1][0].squeeze((0, 1)))), axis=0, times=2),
                                 B.expand_dims(B.concat(*(batch["contexts"][0][1].squeeze((0, 1)), ft[num_layers-1].squeeze((0, 1, 2)))), axis=0, times=2))]
            expaned_contexts.extend([(empty, empty) for _ in range(num_layers-1)])
        else:
            expaned_contexts = batch["contexts"]

        for _ in range(num_samples):

            # Mask context for noisiest layer
            contexts = mask_contexts(batch["contexts"], num_layers-1)
            expaned_contexts = mask_contexts(expaned_contexts, num_layers-1)

            prop_xt_size = 5*og_context_size
            max_xt_size = B.length(batch["xt"][0][0])
            if prop_xt_size < 20:
                prop_xt_size = 20

            if config:
                FONTSIZE = 17
                colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][:]
                fig, axs = plt.subplots(1, 3, sharey=True, figsize=(18, 6))
                plt.subplots_adjust(bottom=0.5)

            for level_index in range(num_layers-1, -1, -1):

                if prop_context and level_index != 0:
                    # Generate proportional targets
                    xt_reshape = batch["xt"][level_index][0].squeeze((0, 1))
                    perm = torch.IntTensor(B.randperm(len(xt_reshape))).to(xt_reshape.device)
                    xt_choice = torch.index_select(xt_reshape, 0, perm[:prop_xt_size])
                    xt_prop = AggregateInput(*((B.expand_dims(xt_choice, axis=0, times=2), i) for i in range(num_layers)))
                    prop_xt_size = prop_xt_size * 2
                    if prop_xt_size > max_xt_size:
                        prop_xt_size = max_xt_size

                if config:
                    l_x = mask_xt(x, level_index)
                    state, pred = model(state,
                                        contexts if level_index != num_layers-1 else expaned_contexts,
                                        l_x)

                if level_index != 0:
                    l_xt = mask_xt(xt_prop if prop_context else batch["xt"], level_index)
                    state, _, _, _, yt = nps.predict(state,
                                                    model,
                                                    contexts if level_index != num_layers-1 else expaned_contexts,
                                                    l_xt,
                                                    num_samples=1,
                                                    batch_size=1)
                    l_yt = yt[level_index].squeeze(0).float()
                    contexts[level_index] = (l_xt[level_index][0], l_yt)

                if config:
                    axs[level_index].scatter(contexts[0][0].squeeze(0).cpu(), contexts[0][1].squeeze(0).cpu(), marker="o", s=25, linewidth=2, label="Original context", color="blue", zorder=2)
                    for j in range(num_layers):
                        if j>level_index:
                            clr = "#217D21" if j==1 else colors[3]
                            axs[level_index].scatter(contexts[j][0].squeeze(0).cpu(), contexts[j][1].squeeze(0).cpu(), label=f"Auxiliary context #{j}", marker="^", c=clr, s=25, linewidth=2)

                    if level_index == 0:
                        axs[level_index].scatter(
                            batch["xt"][level_index][0].cpu(),
                            batch["yt"][level_index].cpu(),
                            marker="x", s=35, linewidth=2, label="Original targets", color=colors[1], zorder=2
                        )

                    # Plot prediction.
                    err = 1.96 * B.sqrt(pred.var[level_index][0, 0])
                    axs[level_index].plot(
                        l_x[level_index][0].squeeze().cpu(),
                        pred.mean[level_index][0, 0].cpu(),
                        label="Prediction",
                        linewidth=2, color="#4BA6FB", linestyle="dashed"
                    )
                    axs[level_index].fill_between(
                        l_x[level_index][0].squeeze().cpu(),
                        pred.mean[level_index][0, 0].cpu() - err.cpu(),
                        pred.mean[level_index][0, 0].cpu() + err.cpu(),
                        color="#4BA6FB",
                        alpha=0.4,
                        linewidth=0,
                        zorder=0,
                    )

                    if level_index != 0:
                        axs[level_index].scatter(l_xt[level_index][0].cpu(), l_yt.cpu(), marker="s", c="black", s=25, label="Prediction sample")
                    else:
                        axs[level_index].scatter(None, None, marker="s", c="black", s=25, label="Prediction sample")

                    axs[level_index].set_xlim(B.min(l_x[level_index][0].cpu()), B.max(l_x[level_index][0].cpu()))

                    axs[level_index].set_axisbelow(True)  # Show grid lines below other elements.
                    axs[level_index].grid(which="major", c="#c0c0c0", alpha=0.5, lw=1)
                    axs[level_index].spines["top"].set_visible(False)
                    axs[level_index].spines["right"].set_visible(False)
                    axs[level_index].spines["bottom"].set_lw(1)
                    axs[level_index].spines["left"].set_lw(1)
                    axs[level_index].xaxis.set_tick_params(width=1)
                    axs[level_index].yaxis.set_tick_params(width=1)
                    if level_index == 0:
                        axs[level_index].tick_params(axis="y", which="major", labelsize=FONTSIZE)
                    axs[level_index].tick_params(axis="x", which="major", labelsize=FONTSIZE)

            if config:
                plt.gca().set_ylim(bottom=-0.5)
                handles, labels = axs[0].get_legend_handles_labels()
                leg = fig.legend(handles, labels, loc="upper center", facecolor="#eeeeee", handletextpad=0.1, edgecolor="#ffffff", framealpha=1, fontsize=FONTSIZE, ncol=6)
                leg.get_frame().set_linewidth(0)
                path=path[:-3]
                plt.tight_layout()
                plt.savefig(f"{path}pdf")
                plt.close()

            config = False
            
            l_xt = mask_xt(batch["xt"], 0)
            state, pred = model(state, contexts, l_xt)

            this_logpdfs = pred.logpdf(B.cast(torch.float32, true_y0t))

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


def split_AR_prediction():
    pass