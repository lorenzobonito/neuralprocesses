import lab as B
import torch

__all__ = ["mask_context"]


def mask_contexts(contexts, batch_size, level_index):
    
    if level_index == 0:
        # If 0, no masking is carried out, and all three context sets are passed on.
        pass
    elif level_index == 1:
        # If 1, y1t is masked, and only y2t and yc are passed as context
        contexts[1] = (B.randn(torch.float32, batch_size, 1, 0), B.randn(torch.float32, batch_size, 1, 0))
    elif level_index == 2:
        # If 2, y1t and y2t are masked, and only yc is passed as context
        contexts[1] = (B.randn(torch.float32, batch_size, 1, 0), B.randn(torch.float32, batch_size, 1, 0))
        contexts[2] = (B.randn(torch.float32, batch_size, 1, 0), B.randn(torch.float32, batch_size, 1, 0))
    
    # Unifying context sets into a single one
    xcontexts = tuple([context[0] for context in contexts])
    ycontexts = tuple([context[1] for context in contexts])

    return [(torch.cat(xcontexts, 2), torch.cat(ycontexts, 2))]
