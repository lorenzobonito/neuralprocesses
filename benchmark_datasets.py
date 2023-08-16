import itertools

import torch
import lab as B
import neuralprocesses.torch as nps
from neuralprocesses.data.noised_sawtooth import NoisedSawtoothGenerator
from neuralprocesses.data.noised_square_wave import NoisedSquareWaveGenerator
from neuralprocesses.data.noised_gp import NoisedGPGenerator
from neuralprocesses.dist.uniform import UniformDiscrete, UniformContinuous
from neuralprocesses.aggregate import Aggregate, AggregateInput

X_RANGE_CONTEXT = (-2, 2)
X_RANGE_TARGET = (-2, 2)


def get_batches(num_context: int, num_batches: int, gen_type: str, target_size: int, config: dict):

    if gen_type.lower() == "noised_sawtooth":
        gen = NoisedSawtoothGenerator(
            torch.float32,
            seed=42,
            noise=0,
            dist_freq=UniformContinuous(2, 4),
            noise_levels=0,
            num_context=UniformDiscrete(num_context, num_context),
            num_target=UniformDiscrete(target_size, target_size),
            **config,
        )
    elif gen_type.lower() == "noised_square_wave":
        gen = NoisedSquareWaveGenerator(
            torch.float32,
            seed=42,
            noise=0,
            noise_levels=0,
            num_context=UniformDiscrete(num_context, num_context),
            num_target=UniformDiscrete(target_size, target_size),
            **config,
        )
    elif gen_type.lower() == "noised_gp":
        gen = NoisedGPGenerator(
            torch.float32,
            seed=42,
            noise=0,
            noise_levels=0,
            num_context=UniformDiscrete(num_context, num_context),
            num_target=UniformDiscrete(target_size, target_size),
            **config,
        )
    else:
        raise ValueError("Selected gen_type has not been implemented.")

    batches = []
    for _ in range(num_batches):
        batches.append(gen.generate_batch())
    
    return batches


if __name__ == "__main__":

    DIM_Y = [1, 3, 4, 5, 6]
    GEN_TYPE = ["noised_sawtooth", "noised_gp", "noised_square_wave"]
    TARGET_SIZE = [50]

    for dim_y, gen_type, target_size in itertools.product(DIM_Y, GEN_TYPE, TARGET_SIZE):

        config = {
                "num_tasks": 1,
                "batch_size": 1,
                "dist_x_context": UniformContinuous(*((X_RANGE_CONTEXT,))),
                "dist_x_target": UniformContinuous(*((X_RANGE_TARGET,))),
                "dim_y": 1,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            }

        batches = []
        for context_size in range(31):
            batches.extend(get_batches(context_size, 10, gen_type, target_size, config))

        empty = B.randn(torch.float32, 1, 1, 0)
        for batch in batches:
            xts = [batch["xt"][0]]
            yts = [batch["yt"][0]]
            for i in range(1, dim_y):
                batch["contexts"].append((empty, empty))
                xts.append((empty, i))
                yts.append(empty)
            batch["xt"] = AggregateInput(*xts)
            batch["yt"] = Aggregate(*yts)

        torch.save(batches, f"benchmark_dataset_{target_size}_targets_{gen_type}_{dim_y}_layers.pt")