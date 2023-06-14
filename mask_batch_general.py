import lab as B
import torch
from neuralprocesses.aggregate import Aggregate, AggregateInput

__all__ = ["mask_batch"]


def mask_batch(in_batch, level_index):

    batch = in_batch.copy()
    batch_size = batch["xt"][0][0].shape[0]
    num_layers = len(batch["xt"])
    empty = B.randn(torch.float32, batch_size, 1, 0)

    # Mask contexts
    for i in range(level_index):
        batch["contexts"][i+1] = (empty, empty)
    
    # Mask xt and yt
    xt, yt = [], []
    for i in range(num_layers):
        if i == level_index:
            xt.append(batch["xt"][i])
            yt.append(batch["yt"][i])
        else:
            xt.append((empty, i))
            yt.append(empty)
    batch["xt"] = AggregateInput(*xt)
    batch["yt"] = Aggregate(*yt)

    return batch
