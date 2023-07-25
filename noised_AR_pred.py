import lab as B
from neuralprocesses.aggregate import Aggregate, AggregateInput
import neuralprocesses.torch as nps
import numpy as np
import torch
import matplotlib.pyplot as plt

from wbml.plot import tweak
from batch_masking import mask_contexts, mask_xt, mask_yt

__all__ = ["joint_AR_prediction", "split_AR_prediction"]


def split_AR_prediction(state, models, batch, num_samples, ar_context, prop_context: bool = False, normalise=True, path=None, config=None):

    if prop_context and ar_context != 0:
        raise ValueError("AR context process not implemented for proportional context.")

    true_y0t = batch["yt"]
    num_layers = len(models)
    batch_size = batch["contexts"][0][0].shape[0]
    empty = B.randn(torch.float32, batch_size, 1, 0)
    float = B.dtype_float(true_y0t)
    float64 = B.promote_dtypes(float, np.float64)
    
    if config:
        try:
            plot_config = config["plot"][1]
        except KeyError:
            return

    with B.on_device(batch["xt"]):
        x = B.linspace(B.dtype(batch["xt"]), *plot_config["range"], 200)
        x = nps.AggregateInput((x[None, None, :], 0))

    og_context_size = B.length(batch["contexts"][0][0])

    with torch.no_grad():

        logpdfs = None

        if og_context_size < ar_context:
            xt_reshape = batch["xt"][0][0].squeeze((0, 1))
            perm = torch.IntTensor(B.randperm(len(xt_reshape))).to(xt_reshape.device)
            xt_choice = torch.index_select(xt_reshape, 0, perm[:ar_context-og_context_size])
            # Line below allows for duplicates, which is not ideal
            # state, xt_choice = B.choice(state, batch["xt"][0][0].squeeze((0, 1)), (ar_context-og_context_size))
            xt_subsample = AggregateInput((B.expand_dims(xt_choice, axis=0, times=2), 0))
            state, _, _, ft, _ = nps.ar_predict(state, models[num_layers-1], batch["contexts"], xt_subsample, num_samples=1, order="random")
            expaned_contexts = [(B.expand_dims(B.concat(*(batch["contexts"][0][0].squeeze((0, 1)), xt_subsample[0][0].squeeze((0, 1)))), axis=0, times=2),
                                 B.expand_dims(B.concat(*(batch["contexts"][0][1].squeeze((0, 1)), ft[0].squeeze((0, 1, 2)))), axis=0, times=2))]
        else:
            expaned_contexts = batch["contexts"]

        for _ in range(num_samples):

            # Re-format context for noisiest layer
            contexts = batch["contexts"]
            for _ in range(num_layers-1):
                contexts.append((empty, empty))
                expaned_contexts.append((empty, empty))

            prop_xt_size = 5*og_context_size
            max_xt_size = B.length(batch["xt"][0][0])
            if prop_xt_size < 20:
                prop_xt_size = 20

            if config:
                plt.figure(figsize=(8, 6 * num_layers))
                colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][2:]

            for level_index in range(num_layers-1, -1, -1):

                if prop_context and level_index != 0:
                    # Generate proportional targets
                    xt_reshape = batch["xt"][0][0].squeeze((0, 1))
                    perm = torch.IntTensor(B.randperm(len(xt_reshape))).to(xt_reshape.device)
                    xt_choice = torch.index_select(xt_reshape, 0, perm[:prop_xt_size])
                    xt_prop = AggregateInput((B.expand_dims(xt_choice, axis=0, times=2), 0))
                    prop_xt_size = prop_xt_size * 2
                    if prop_xt_size > max_xt_size:
                        prop_xt_size = max_xt_size

                if config:
                    state, pred = models[level_index](state,
                                                      contexts if level_index != num_layers-1 else expaned_contexts,
                                                      x)

                if level_index != 0:
                    state, _, _, _, yt = nps.predict(state,
                                                    models[level_index],
                                                    contexts if level_index != num_layers-1 else expaned_contexts,
                                                    xt_prop if prop_context else batch["xt"],
                                                    num_samples=1,
                                                    batch_size=1)
                    l_yt = yt[0].squeeze(0).float()
                    contexts[level_index] = (xt_prop[0][0] if prop_context else batch["xt"][0][0], l_yt)

                if config:
                    plt.subplot(num_layers, 1, level_index+1)
                    if level_index == num_layers-1:
                        plt.scatter(expaned_contexts[0][0].squeeze(0), expaned_contexts[0][1].squeeze(0), label="AR Context", style="train", c="magenta", s=20)
                    plt.scatter(contexts[0][0].squeeze(0), contexts[0][1].squeeze(0), label="Original Context", style="train", c="blue", s=20)
                    for j in range(num_layers):
                        if j>level_index:
                            plt.scatter(contexts[j][0].squeeze(0), contexts[j][1].squeeze(0), label=f"Auxiliary Context {j}", style="train", marker="^", c=colors[j-1], s=10)

                    if level_index == 0:
                        plt.scatter(
                            batch["xt"][0][0],
                            batch["yt"][0],
                            label="Target",
                            style="test",
                            s=20,
                        )

                    # Plot prediction.
                    err = 1.96 * B.sqrt(pred.var[0][0, 0])
                    plt.plot(
                        x[0][0],
                        pred.mean[0][0, 0],
                        label="Prediction",
                        style="pred",
                    )
                    plt.fill_between(
                        x[0][0],
                        pred.mean[0][0, 0] - err,
                        pred.mean[0][0, 0] + err,
                        style="pred",
                    )

                    if level_index != 0:
                        plt.scatter(xt_prop[0][0] if prop_context else batch["xt"][0][0], l_yt, marker="s", c="black", s=10, label="Prediction sample")

                    for x_axvline in plot_config["axvline"]:
                        plt.axvline(x_axvline, c="k", ls="--", lw=0.5)

                    plt.xlim(B.min(x[0][0]), B.max(x[0][0]))
                    tweak()

            if config:
                plt.savefig(path)
                plt.close()

            config = False
            
            state, pred = models[level_index](state, contexts, batch["xt"])
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
                plt.figure(figsize=(8, 6 * num_layers))
                colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][2:]

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
                    plt.subplot(num_layers, 1, level_index+1)
                    if level_index == num_layers-1 and ar_context != 0:
                        plt.scatter(expaned_contexts[0][0].squeeze(0), expaned_contexts[0][1].squeeze(0), label="AR Context", style="train", c="magenta", s=20)
                    plt.scatter(contexts[0][0].squeeze(0), contexts[0][1].squeeze(0), label="Original Context", style="train", c="blue", s=20)
                    for j in range(num_layers):
                        if j>level_index:
                            plt.scatter(contexts[j][0].squeeze(0), contexts[j][1].squeeze(0), label=f"Auxiliary Context {j}", style="train", marker="^", c=colors[j-1], s=10)

                    if level_index == 0:
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
