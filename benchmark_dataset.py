import torch
import lab as B
import neuralprocesses.torch as nps
from neuralprocesses.data.noised_sawtooth import NoisedSawtoothGenerator
from neuralprocesses.data.noised_square_wave import NoisedSquareWaveGenerator
from neuralprocesses.data.noised_gp import NoisedGPGenerator
from neuralprocesses.dist.uniform import UniformDiscrete, UniformContinuous
from neuralprocesses.aggregate import Aggregate, AggregateInput

DIM_X = 1
X_RANGE_CONTEXT = (-2, 2)
X_RANGE_TARGET = (-2, 2)


def get_batches(num_context: int, num_batches: int, gen_type: str, config: dict):

    if gen_type.lower() == "noised_sawtooth":
        gen = NoisedSawtoothGenerator(
            torch.float32,
            seed=42,
            noise=0,
            dist_freq=UniformContinuous(2 / B.sqrt(DIM_X), 4 / B.sqrt(DIM_X)),
            noise_levels=0,
            num_context=UniformDiscrete(num_context, num_context),
            num_target=UniformDiscrete(5 * DIM_X, 50 * DIM_X),
            **config,
        )
    elif gen_type.lower() == "noised_square_wave":
        gen = NoisedSquareWaveGenerator(
            torch.float32,
            seed=42,
            noise=0,
            noise_levels=0,
            num_context=UniformDiscrete(num_context, num_context),
            num_target=UniformDiscrete(5 * DIM_X, 50 * DIM_X),
            **config,
        )
    elif gen_type.lower() == "noised_gp":
        gen = NoisedGPGenerator(
            torch.float32,
            seed=42,
            noise=0,
            noise_levels=0,
            num_context=UniformDiscrete(num_context, num_context),
            num_target=UniformDiscrete(5 * DIM_X, 50 * DIM_X),
            **config,
        )
    else:
        raise ValueError("Selected gen_type has not been implemented.")

    batches = []
    for _ in range(num_batches):
        batches.append(gen.generate_batch())
    
    return batches


if __name__ == "__main__":

    DIM_Y = 3
    # GEN_TYPE = "noised_sawtooth" 
    # GEN_TYPE = "noised_square_wave"
    GEN_TYPE = "noised_gp"

    config = {
            "num_tasks": 1,
            "batch_size": 1,
            "dist_x_context": UniformContinuous(*((X_RANGE_CONTEXT,) * DIM_X)),
            "dist_x_target": UniformContinuous(*((X_RANGE_TARGET,) * DIM_X)),
            "dim_y": 1,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }

    batches = []
    for context_size in range(31):
        batches.extend(get_batches(context_size, 10, GEN_TYPE, config))

    empty = B.randn(torch.float32, 1, 1, 0)
    for batch in batches:
        xts = [batch["xt"][0]]
        yts = [batch["yt"][0]]
        for i in range(1, DIM_Y):
            batch["contexts"].append((empty, empty))
            xts.append((empty, i))
            yts.append(empty)
        batch["xt"] = AggregateInput(*xts)
        batch["yt"] = Aggregate(*yts)

    torch.save(batches, f"benchmark_dataset_varTarg_{GEN_TYPE}_{DIM_Y}_layers.pt")