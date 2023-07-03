import argparse
from copy import deepcopy
import json
from multiprocessing import Process
import os
import sys
import warnings
from functools import partial

import experiment as exp
import lab as B
import neuralprocesses.torch as nps
import numpy as np
import torch
import wbml.out as out
from matrix.util import ToDenseWarning
from wbml.experiment import WorkingDirectory
from batch_masking import mask_batch
from noised_AR_pred import split_AR_prediction, joint_AR_prediction
from training_dynamics import split_training_dynamics, joint_training_dynamics

__all__ = ["main"]

warnings.filterwarnings("ignore", category=ToDenseWarning)
    

def train(state, model, opt, objective, gen, *, fix_noise, level_index: int):
    """Train for an epoch."""

    # Preliminary setup
    if level_index is not None:
        split = True
        vals = []
    else:
        split = False
        levels = tuple([level for level in range(gen.noise_levels+1)])
        vals = {}
        for level in levels:
            vals[level] = []
    
    for batch in gen.epoch():
        if not split:
            level_index = np.random.choice(levels)
        batch = mask_batch(batch, level_index, split)
        state, obj = objective(
            state,
            model,
            batch["contexts"],
            batch["xt"],
            batch["yt"],
            fix_noise=fix_noise,
        )
        if split:
            vals.append(B.to_numpy(obj))
        else:
            vals[level_index].append(B.to_numpy(obj))
        val = -B.mean(obj)
        opt.zero_grad(set_to_none=True)
        val.backward()
        opt.step()

    if split:
        vals = B.concat(*vals)
        out.kv("Loglik (T)", exp.with_err(vals, and_lower=True))
        return state, B.mean(vals) - 1.96 * B.std(vals) / B.sqrt(len(vals))
    else:
        vals = dict((k, B.concat(*v)) for k, v in vals.items())
        for level in levels:
            out.kv(f"Loglik (T, {level})", exp.with_err(vals[level], and_lower=True))
        vals = np.array([B.mean(B.concat(vals[level])) - 1.96 * B.std(B.concat(vals[level])) / B.sqrt(len(B.concat(vals[level]))) for level in levels])
        return state, vals


def eval(state, model, objective, gen, level_index: int):
    """Perform evaluation."""
    with torch.no_grad():

        # Preliminary setup
        if level_index is not None:
            split = True
            vals = []
        else:
            split = False
            levels = tuple([level for level in range(gen.noise_levels+1)])
            vals = {}
            for level in levels:
                vals[level] = []
                
        for batch in gen.epoch():
            if not split:
                level_index = np.random.choice(levels)
            batch = mask_batch(batch, level_index, split)
            state, obj = objective(
                state,
                model,
                batch["contexts"],
                batch["xt"],
                batch["yt"],
            )
            if split:
                vals.append(B.to_numpy(obj))
            else:
                vals[level_index].append(B.to_numpy(obj))

        if split:
            vals = B.concat(*vals)
            out.kv("Loglik (V)", exp.with_err(vals, and_lower=True))
            return state, B.mean(vals) - 1.96 * B.std(vals) / B.sqrt(len(vals))
        else:
            vals = dict((k, B.concat(*v)) for k, v in vals.items())
            for level in levels:
                out.kv(f"Loglik (V, {level})", exp.with_err(vals[level], and_lower=True))
            vals = np.array([B.mean(B.concat(vals[level])) - 1.96 * B.std(B.concat(vals[level])) / B.sqrt(len(B.concat(vals[level]))) for level in levels])
            return state, vals


def main(**kw_args):
    # Setup arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, nargs="*", default=["_experiments"])
    parser.add_argument("--subdir", type=str, nargs="*")
    parser.add_argument("--device", type=str)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--dim-x", type=int, default=1)
    parser.add_argument("--dim-y", type=int, default=1)

    # Only passed if split model is being trained
    parser.add_argument("--model-index", type=int, default=None)
    parser.add_argument("--noise-levels", type=int, default=None)

    parser.add_argument("--num-unet-channels", type=int, default=6)
    parser.add_argument("--size-unet-channels", type=int, default=64)
    parser.add_argument("--unet-kernels", type=int, default=5)
    parser.add_argument("--ar-samples", type=int, default=100)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--rate", type=float)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--arch",
        choices=[
            "unet",
            "unet-sep",
            "unet-res",
            "unet-res-sep",
            "conv",
            "conv-sep",
            "conv-res",
            "conv-res-sep",
        ],
        default="unet",
    )
    parser.add_argument(
        "--data",
        choices=exp.data,
        default="eq",
    )
    parser.add_argument("--mean-diff", type=float, default=None)
    parser.add_argument("--objective", choices=["loglik", "elbo"], default="loglik")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--resume-at-epoch", type=int)
    parser.add_argument("--train-fast", action="store_true")
    parser.add_argument("--check-completed", action="store_true")
    parser.add_argument("--unnormalised", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--evaluate-last", action="store_true")
    parser.add_argument("--evaluate-fast", action="store_true")
    parser.add_argument("--evaluate-num-plots", type=int, default=5)
    parser.add_argument("--evaluate-objective", type=str, default="loglik")
    parser.add_argument("--evaluate-num-samples", type=int, default=512)
    parser.add_argument("--evaluate-batch-size", type=int, default=8)
    parser.add_argument("--no-action", action="store_true")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--experiment-setting", type=str, nargs="*")
    parser.add_argument(
        "--eeg-mode",
        type=str,
        choices=["random", "interpolation", "forecasting", "reconstruction"],
    )
    parser.add_argument("--patch", type=str)

    if kw_args:
        # Load the arguments from the keyword arguments passed to the function.
        # Carefully convert these to command line arguments.
        args = parser.parse_args(
            sum(
                [
                    ["--" + k.replace("_", "-")] + ([str(v)] if v is not True else [])
                    for k, v in kw_args.items()
                ],
                [],
            )
        )
    else:
        args = parser.parse_args()

    def patch_model(d):
        """Patch a loaded model.

        Args:
            d (dict): Output of :func:`torch.load`.

        Returns:
            dict: `d`, but patched.
        """
        if args.patch:
            with out.Section("Patching loaded model"):
                # Loop over patches.
                for patch in args.patch.strip().split(";"):
                    base_from, base_to = patch.split(":")

                    # Try to apply the patch.
                    applied_patch = False
                    for k in list(d["weights"].keys()):
                        if k.startswith(base_from):
                            applied_patch = True
                            tail = k[len(base_from) :]
                            d["weights"][base_to + tail] = d["weights"][k]
                            del d["weights"][k]

                    # Report whether the patch was applied.
                    if applied_patch:
                        out.out(f'Applied patch "{patch}".')
                    else:
                        out.out(f'Did not apply patch "{patch}".')
        return d

    # Remove the dimensionality specification if the experiment doesn't need it.
    if not exp.data[args.data]["requires_dim_x"]:
        del args.dim_x
    if not exp.data[args.data]["requires_dim_y"]:
        del args.dim_y

    # Ensure that `args.experiment_setting` is always a list.
    if not args.experiment_setting:
        args.experiment_setting = []

    # Determine settings for the setup of the script.
    observe = False
    if args.check_completed or args.no_action or args.load:
        observe = True

    data_dir = args.data if args.mean_diff is None else f"{args.data}-{args.mean_diff}"
    data_dir = data_dir if args.eeg_mode is None else f"{args.data}-{args.eeg_mode}"

    # Setup script.
    if not observe:
        out.report_time = True
    if args.model_index != -1:
        wd_train = WorkingDirectory(
            *args.root,
            *(args.subdir or ()),
            data_dir,
            "split" if args.model_index is not None else "joint",
            f"{args.noise_levels+1}_layers" if args.model_index is not None else f"{args.dim_y}_layers",
            "convcnp",
            *((args.arch,) if hasattr(args, "arch") else ()),
            f"s{args.size_unet_channels}_n{args.num_unet_channels}_k{args.unet_kernels}",
            f"{args.epochs}_epochs",
            "train",
            str(args.model_index) if args.model_index is not None else "",
            log=f"log_train.txt" if not args.evaluate else None,
            diff=f"diff_train.txt" if not args.evaluate else None,
            observe=observe,
        )

    # Determine which device to use. Try to use a GPU if one is available.
    if args.device:
        device = args.device
    elif args.gpu is not None:
        device = f"cuda:{args.gpu}"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    B.set_global_device(device)
    # Maintain an explicit random state through the execution.
    state = B.create_random_state(torch.float32, seed=0)

    # General config.
    if args.dim_y == 1:
        noise_levels = args.noise_levels
    else:
        noise_levels = args.dim_y - 1
    config = {
        "default": {
            "epochs": None,
            "rate": None,
            "also_ar": False,
        },
        "epsilon": 1e-8,
        "epsilon_start": 1e-2,
        "cholesky_retry_factor": 1e6,
        "fix_noise": None,
        "fix_noise_epochs": 3,
        "width": 256,
        "dim_embedding": 256,
        "enc_same": False,
        "num_heads": 8,
        "num_layers": 6,
        "unet_channels": (args.size_unet_channels,) * args.num_unet_channels,
        "unet_kernels": args.unet_kernels,
        # "unet_strides": (1,) + (2,) * (args.num_unet_channels-1),
        "unet_strides": (2,) * (args.num_unet_channels),
        "conv_channels": 64,
        "encoder_scales": None,
        "fullconvgnp_kernel_factor": 2,
        "mean_diff": args.mean_diff,
        # Performance of the ConvGNP is sensitive to this parameter. Moreover, it
        # doesn't make sense to set it to a value higher of the last hidden layer of
        # the CNN architecture. We therefore set it to 64.
        "num_basis_functions": 64,
        "eeg_mode": args.eeg_mode,
        "noise_levels": noise_levels,
    }

    # Setup data generators for training and for evaluation.
    gen_train, gen_cv, gens_eval = exp.data[args.data]["setup"](
        args,
        config,
        num_tasks_train=2**6 if args.train_fast else 2**14,
        num_tasks_cv=2**6 if args.train_fast else 2**12,
        num_tasks_eval=2**6 if args.evaluate_fast else 2**12,
        device=device,
    )

    # Apply defaults for the number of epochs and the learning rate. The experiment
    # is allowed to adjust these.
    args.epochs = args.epochs or config["default"]["epochs"] or 100
    args.rate = args.rate or config["default"]["rate"] or 3e-4

    # Check if a run has completed.
    if args.check_completed:
        if os.path.exists(wd_train.file("model-last.torch")):
            d = patch_model(torch.load(wd_train.file("model-last.torch"), map_location="cpu"))
            if d["epoch"] >= args.epochs - 1:
                out.out("Completed!")
                sys.exit(0)
        out.out("Not completed.")
        sys.exit(1)

    # Set the regularisation based on the experiment settings.
    B.epsilon = config["epsilon"]
    B.cholesky_retry_factor = config["cholesky_retry_factor"]

    if "model" in config:
        # See if the experiment constructed the particular flavour of the model already.
        model = config["model"]
    else:
        # ConvCNP
        model = nps.construct_convgnp(
            points_per_unit=config["points_per_unit"],
            dim_x=config["dim_x"],
            dim_yc=(1,) * (noise_levels+1),
            dim_yt=config["dim_y"],
            likelihood="het",
            conv_arch=args.arch,
            unet_channels=config["unet_channels"],
            unet_kernels=config["unet_kernels"],
            unet_strides=config["unet_strides"],
            conv_channels=config["conv_channels"],
            conv_layers=config["num_layers"],
            conv_receptive_field=config["conv_receptive_field"],
            margin=config["margin"],
            encoder_scales=config["encoder_scales"],
            transform=config["transform"],
        )

    # Settings specific for the model:
    if config["fix_noise"] is None:
            config["fix_noise"] = False

    # Ensure that the model is on the GPU and print the setup.
    model = model.to(device)
    if not args.load and not args.evaluate:
        out.kv(
            "Arguments",
            {
                attr: getattr(args, attr)
                for attr in args.__dir__()
                if not attr.startswith("_")
            },
        )
        out.kv(
            "Config", {k: "<custom>" if k == "model" else v for k, v in config.items()}
        )
        out.kv("Number of parameters", nps.num_params(model))


    objective = partial(
        nps.loglik,
        num_samples=args.num_samples,
        normalise=not args.unnormalised,
    )
    objective_cv = partial(
        nps.loglik,
        num_samples=args.num_samples,
        normalise=not args.unnormalised,
    )

    # See if the point was to just load everything.
    if args.load:
        return {
            "wd": wd_train,
            "gen_train": gen_train,
            "gen_cv": gen_cv,
            "gens_eval": gens_eval,
            "model": model,
        }

    # The user can just want to see some statistics about the model.
    if args.no_action:
        exit()

    if args.evaluate:

        wd_eval = WorkingDirectory(
            *args.root,
            *(args.subdir or ()),
            data_dir,
            "split" if args.model_index is not None else "joint",
            f"{args.noise_levels+1}_layers" if args.model_index is not None else f"{args.dim_y}_layers",
            "convcnp",
            *((args.arch,) if hasattr(args, "arch") else ()),
            f"s{args.size_unet_channels}_n{args.num_unet_channels}_k{args.unet_kernels}",
            f"{args.epochs}_epochs",
            "eval",
            str(args.ar_samples),
            log=f"log_eval.txt",
            diff=f"diff_eval.txt",
            observe=observe,
        )

        os.makedirs(os.path.join(wd_eval.file(), "images"), exist_ok=True)

        if not args.load:
            out.kv(
                "Arguments",
                {
                    attr: getattr(args, attr)
                    for attr in args.__dir__()
                    if not attr.startswith("_")
                },
            )
            out.kv(
                "Config", {k: "<custom>" if k == "model" else v for k, v in config.items()}
            )
            out.kv("Number of parameters", nps.num_params(model))

        # Perform evaluation.
        if args.evaluate_last:
            name = "model-last.torch"
        else:
            name = "model-best.torch"
        if args.model_index is not None:
            if args.model_index != -1:
                raise ValueError("model_index parameter must be set to -1 when evaluating.")
            models = []
            for index in range(args.noise_levels+1):
                # ConvCNP
                model = nps.construct_convgnp(
                    points_per_unit=config["points_per_unit"],
                    dim_x=config["dim_x"],
                    dim_yc=(1,) * (noise_levels+1),
                    dim_yt=config["dim_y"],
                    likelihood="het",
                    conv_arch=args.arch,
                    unet_channels=config["unet_channels"],
                    unet_kernels=config["unet_kernels"],
                    unet_strides=config["unet_strides"],
                    conv_channels=config["conv_channels"],
                    conv_layers=config["num_layers"],
                    conv_receptive_field=config["conv_receptive_field"],
                    margin=config["margin"],
                    encoder_scales=config["encoder_scales"],
                    transform=config["transform"],
                )
                wd_load = WorkingDirectory(
                    *args.root,
                    *(args.subdir or ()),
                    data_dir,
                    "split" if args.model_index is not None else "joint",
                    f"{args.noise_levels+1}_layers" if args.model_index is not None else f"{args.dim_y}_layers",
                    "convcnp",
                    *((args.arch,) if hasattr(args, "arch") else ()),
                    f"s{args.size_unet_channels}_n{args.num_unet_channels}_k{args.unet_kernels}",
                    f"{args.epochs}_epochs",
                    "train",
                    str(index),
                    log=f"log_train.txt" if not args.evaluate else None,
                    diff=f"diff_train.txt" if not args.evaluate else None,
                    observe=observe,
                )
                model.load_state_dict(patch_model(torch.load(wd_load.file(name), map_location=device))["weights"])
                model.to(device)
                models.append(model)
        else:
            model.load_state_dict(patch_model(torch.load(wd_train.file(name), map_location=device))["weights"])

        # Load different context sets
        dataset = torch.load(f"benchmark_datasets/benchmark_dataset_{args.data}_{args.dim_y}_layers.pt", map_location=device)
        
        # Evaluate model predictions over context sets
        logliks = []
        json_data = {}
        for idx, batch in enumerate(dataset):
            if args.model_index is not None:
                state, loglik = split_AR_prediction(state, models, batch, num_samples=args.ar_samples, path=wd_eval.file(f"images/noised_AR_pred-{idx + 1:03d}.pdf"), config=config)
            else:
                state, loglik = joint_AR_prediction(state, model, batch, num_samples=args.ar_samples, path=wd_eval.file(f"images/noised_AR_pred-{idx + 1:03d}.pdf"), config=config)
            logliks.append(loglik)
            json_data[idx] = (batch["contexts"][0][0].numel(), loglik.item())
            out.kv(f"Dataset {idx}", (str(batch["contexts"][0][0].numel()), *loglik))
            with open(wd_eval.file("logliks.json"), "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)

        logliks = B.concat(*logliks)
        out.kv("Loglik (P)", exp.with_err(logliks, and_lower=True))

    else:
        # Perform training. First, check if we want to resume training.
        start = 0
        if args.resume_at_epoch:
            start = args.resume_at_epoch - 1
            d_last = patch_model(
                torch.load(wd_train.file("model-last.torch"), map_location=device)
            )
            d_best = patch_model(
                torch.load(wd_train.file("model-best.torch"), map_location=device)
            )
            model.load_state_dict(d_last["weights"])
            best_eval_lik = d_best["objective"]
        else:
            if args.model_index is not None:
                # Split model
                best_eval_lik = -np.inf
            else:
                best_eval_lik = np.ones((1, args.dim_y))*-np.inf

        # Setup training loop.
        opt = torch.optim.Adam(model.parameters(), args.rate)

        # Set regularisation high for the first epochs.
        original_epsilon = B.epsilon
        B.epsilon = config["epsilon_start"]

        for i in range(start, args.epochs):
            with out.Section(f"Epoch {i + 1}"):
                # Set regularisation to normal after the first epoch.
                if i > 0:
                    B.epsilon = original_epsilon

                # Checkpoint at regular intervals if specified
                if args.checkpoint_every is not None and i % args.checkpoint_every == 0:
                    out.out("Checkpointing...")
                    torch.save(
                        {
                            "weights": model.state_dict(),
                            "epoch": i + 1,
                        },
                        wd_train.file(f"model-epoch-{i+1}.torch"),
                    )

                # Perform an epoch.
                if config["fix_noise"] and i < config["fix_noise_epochs"]:
                    fix_noise = 1e-4
                else:
                    fix_noise = None
                state, _ = train(
                    state,
                    model,
                    opt,
                    objective,
                    gen_train,
                    fix_noise=fix_noise,
                    level_index=args.model_index
                )

                # The epoch is done. Now evaluate.
                state, vals = eval(state, model, objective_cv, gen_cv(), level_index=args.model_index)

                # Save current model.
                torch.save(
                    {
                        "weights": model.state_dict(),
                        "objective": vals,
                        "epoch": i + 1,
                    },
                    wd_train.file(f"model-last.torch"),
                )

                improved = vals > best_eval_lik
                # Check if the model is the new best. If so, save it.
                if improved.all():
                    out.out("New best model!")
                    best_eval_lik = vals
                    torch.save(
                        {
                            "weights": model.state_dict(),
                            "objective": vals,
                            "epoch": i + 1,
                        },
                        wd_train.file(f"model-best.torch"),
                    )

                # Creating folder for output images
                os.makedirs(os.path.join(wd_train.file(), "images"), exist_ok=True)
                os.makedirs(os.path.join(wd_train.file(), "images", "outputs"), exist_ok=True)
                
                # Visualise a few predictions by the model every 10 epochs.
                if i % 10 == 0 or i == args.epochs-1:
                    gen = gen_cv()
                    if args.model_index is not None:
                        for j in range(5):
                            exp.visualise_noised_split(
                                model,
                                gen,
                                path=wd_train.file(f"images/outputs/epoch-{i + 1:03d}-{j + 1}.pdf"),
                                config=config,
                                model_index=args.model_index
                            )
                    else:
                        for j in range(5):
                            exp.visualise_noised_joint(
                                model,
                                gen,
                                path=wd_train.file(f"images/outputs/epoch-{i + 1:03d}-{j + 1}.pdf"),
                                config=config,
                            )
            
        # Plot training dynamics
        os.makedirs(os.path.join(wd_train.file(), "images", "dynamics"), exist_ok=True)
        if args.model_index is not None:
            split_training_dynamics(wd_train)
        else:
            joint_training_dynamics(wd_train)


if __name__ == "__main__":

    # For joint model, set dim_y = LEVELS.
    # For split model, set noise_levels = LEVELS-1 and model_index \in {0, ..., LEVELS-1} in turn. Use model_index = -1 for evaluation.

    LEVELS = 3

    train_procs = []
    for index in range(LEVELS):
        proc = Process(target=main, 
                       kwargs={"data":"noised_sawtooth",
                               "epochs":500,
                               "noise_levels":LEVELS-1,
                               "model_index":index,
                               "num_unet_channels":12,
                               "size_unet_channels":80,
                               "gpu":1})
        train_procs.append(proc)
        proc.start()
    for proc in train_procs:
        proc.join()

    eval_procs = []
    for ar_samples in [100, 1000]:
        proc = Process(target=main,
                       kwargs={"data":"noised_sawtooth",
                               "epochs":500,
                               "noise_levels":LEVELS-1,
                               "model_index":-1,
                               "evaluate":True,
                               "ar_samples":ar_samples,
                               "num_unet_channels":12,
                               "size_unet_channels":80,
                               "gpu":1})
        eval_procs.append(proc)
        proc.start()
    for proc in eval_procs:
        proc.join()

    # main(data="noised_sawtooth", dim_y=3, epochs=10)
