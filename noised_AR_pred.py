import lab as B
import neuralprocesses.torch as nps
import numpy as np
import torch

__all__ = ["generate_AR_prediction"]

def generate_AR_prediction(state, model, gen, num_samples):

    with torch.no_grad():

        for _ in range(num_samples):

            batch = gen.generate_batch()
            print(batch["contexts"])

        #     state, pred = model(
        #     state,
        #     contexts,
        #     xt,
        #     num_samples=this_num_samples,
        #     dtype_enc_sample=float,
        #     dtype_lik=float64,
        #     **kw_args,
        # )
        # pred = fix_noise_in_pred(pred, fix_noise)


            print(_)
            loglik=0

    return state, loglik

