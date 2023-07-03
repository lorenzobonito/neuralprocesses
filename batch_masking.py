import lab as B
import torch
from neuralprocesses.aggregate import Aggregate, AggregateInput

__all__ = ["mask_batch", "mask_contexts", "mask_xt", "mask_yt"]


def mask_contexts(in_contexts, level_index):

    contexts = in_contexts.copy()
    batch_size = contexts[0][0].shape[0]
    empty = B.randn(torch.float32, batch_size, 1, 0)

    # Mask contexts
    for i in range(level_index):
        contexts[i+1] = (empty, empty)

    return contexts


def mask_xt(in_xt, level_index):

    batch_size = in_xt[0][0].shape[0]
    num_layers = len(in_xt)
    empty = B.randn(torch.float32, batch_size, 1, 0)
    
    # Mask xt
    xt = []
    for i in range(num_layers):
        if i == level_index:
            xt.append(in_xt[i])
        else:
            xt.append((empty, i))
    xt = AggregateInput(*xt)

    return xt


def mask_yt(in_yt, level_index):

    batch_size = in_yt[0].shape[0]
    num_layers = len(in_yt)
    empty = B.randn(torch.float32, batch_size, 1, 0)
    
    # Mask xt and yt
    yt = []
    for i in range(num_layers):
        if i == level_index:
            yt.append(in_yt[i])
        else:
            yt.append(empty)
    yt = Aggregate(*yt)

    return yt


def mask_batch(in_batch, level_index, split):

    batch = in_batch.copy()

    if split:
        batch["contexts"] = mask_contexts(batch["contexts"], level_index)
        batch["xt"] = AggregateInput((batch["xt"][level_index][0], 0))
        batch["yt"] = Aggregate(batch["yt"][level_index])
    else:
        batch["contexts"] = mask_contexts(batch["contexts"], level_index)
        batch["xt"] = mask_xt(batch["xt"], level_index)
        batch["yt"] = mask_yt(batch["yt"], level_index)

    return batch
