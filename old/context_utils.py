import lab as B
import torch

__all__ = ["mask_contexts"]


def mask_contexts(context, batch_size, level_index):

    if level_index == 0:
        # If 0, no masking is carried out, and all three context sets are passed on.
        return context
    elif level_index == 1:
        # # If 1, y1t is masked, and only y2t and yc are passed as context
        context[1] = (B.randn(torch.float32, batch_size, 1, 0), B.randn(torch.float32, batch_size, 1, 0))
        return context
    elif level_index == 2:
        # If 2, y1t and y2t are masked, and only yc is passed as context
        context[1] = (B.randn(torch.float32, batch_size, 1, 0), B.randn(torch.float32, batch_size, 1, 0))
        context[2] = (B.randn(torch.float32, batch_size, 1, 0), B.randn(torch.float32, batch_size, 1, 0))
        return context