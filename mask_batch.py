import lab as B
import torch
from neuralprocesses.aggregate import Aggregate, AggregateInput

__all__ = ["mask_batch"]


def mask_batch(in_batch, level_index):

    batch = in_batch.copy()
    batch_size = batch["contexts"][0][0].shape[0]
    empty = B.randn(torch.float32, batch_size, 1, 0)

    if level_index == 0:
        # If 0, no masking is carried out, and all three context sets are passed on.
        batch["xt"] = AggregateInput(*(batch["xt"][0], (empty, 1), (empty, 2)))
        batch["yt"] = Aggregate(*(batch["yt"][0], empty, empty))
    elif level_index == 1:
        # # If 1, y1t is masked, and only y2t and yc are passed as context
        batch["contexts"][1] = (empty, empty)
        batch["xt"] = AggregateInput(*((empty, 0), batch["xt"][1], (empty, 2)))
        batch["yt"] = Aggregate(*(empty, batch["yt"][1], empty))
    elif level_index == 2:
        # If 2, y1t and y2t are masked, and only yc is passed as context
        batch["contexts"][1] = (empty, empty)
        batch["contexts"][2] = (empty, empty)
        batch["xt"] = AggregateInput(*((empty, 0), (empty, 1), batch["xt"][2]))
        batch["yt"] = Aggregate(*(empty, empty, batch["yt"][2]))

    return batch
