import argparse
import os
import sys
import time
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
from noised_AR_pred_general import generate_AR_prediction

__all__ = ["main"]

warnings.filterwarnings("ignore", category=ToDenseWarning)
    

def train(state, model, opt, objective, gen, *, fix_noise):
    """Train for an epoch."""

    # Preliminary setup
    levels = tuple([level for level in range(gen.noise_levels+1)])
    vals = {}
    for level in levels:
        vals[level] = []
    
    for batch in gen.epoch():
        level_index = np.random.choice(levels)
        batch = mask_batch(batch, level_index)
        state, obj = objective(
            state,
            model,
            batch["contexts"],
            batch["xt"],
            batch["yt"],
            fix_noise=fix_noise,
        )
        vals[level_index].append(B.to_numpy(obj))
        val = -B.mean(obj)
        opt.zero_grad(set_to_none=True)
        val.backward()
        opt.step()

    vals = dict((k, B.concat(*v)) for k, v in vals.items())
    for level in levels:
        out.kv(f"Loglik (T, {level})", exp.with_err(vals[level], and_lower=True))
    vals = B.concat(*vals.values())
    return state, B.mean(vals) - 1.96 * B.std(vals) / B.sqrt(len(vals))


def eval(state, model, objective, gen):
    """Perform evaluation."""
    with torch.no_grad():

        # Preliminary setup
        levels = tuple([level for level in range(gen.noise_levels+1)])
        vals = {}
        for level in levels:
            vals[level] = []
            
        for batch in gen.epoch():
            level_index = np.random.choice(levels)
            batch = mask_batch(batch, level_index)
            state, obj = objective(
                state,
                model,
                batch["contexts"],
                batch["xt"],
                batch["yt"],
            )
            vals[level_index].append(B.to_numpy(obj))

        vals = dict((k, B.concat(*v)) for k, v in vals.items())
        for level in levels:
            out.kv(f"Loglik (V, {level})", exp.with_err(vals[level], and_lower=True))
        vals = B.concat(*vals.values())
        return state, B.mean(vals) - 1.96 * B.std(vals) / B.sqrt(len(vals))


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
    # parser.add_argument("--noise_levels", type=int, default=2)
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
    parser.add_argument("--objective", choices=["loglik", "sl_loglik"], default="sl_loglik")
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
    suffix = ""
    observe = False
    if args.check_completed or args.no_action or args.load:
        observe = True
    elif args.evaluate:
        suffix = "_evaluate"
    else:
        # The default is training.
        suffix = "_train"

    data_dir = args.data if args.mean_diff is None else f"{args.data}-{args.mean_diff}"
    data_dir = data_dir if args.eeg_mode is None else f"{args.data}-{args.eeg_mode}"

    # Setup script.
    if not observe:
        out.report_time = True
    wd = WorkingDirectory(
        *args.root,
        *(args.subdir or ()),
        data_dir,
        *((f"x{args.dim_x}_y{args.dim_y}",) if hasattr(args, "dim_x") else ()),
        "convcnp",
        *((args.arch,) if hasattr(args, "arch") else ()),
        args.objective,
        str(args.epochs),
        log=f"log{suffix}.txt",
        diff=f"diff{suffix}.txt",
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
        "unet_channels": (64,) * 6,
        "unet_strides": (1,) + (2,) * 5,
        "conv_channels": 64,
        "encoder_scales": None,
        "fullconvgnp_kernel_factor": 2,
        "mean_diff": args.mean_diff,
        # Performance of the ConvGNP is sensitive to this parameter. Moreover, it
        # doesn't make sense to set it to a value higher of the last hidden layer of
        # the CNN architecture. We therefore set it to 64.
        "num_basis_functions": 64,
        "eeg_mode": args.eeg_mode,
        "noise_levels": args.dim_y - 1,
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
        if os.path.exists(wd.file("model-last.torch")):
            d = patch_model(torch.load(wd.file("model-last.torch"), map_location="cpu"))
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
            dim_yc=(1,) * config["dim_y"],
            dim_yt=config["dim_y"],
            likelihood="het",
            conv_arch=args.arch,
            unet_channels=config["unet_channels"],
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
    objectives_eval = [
        (
            "Loglik",
            partial(
                nps.loglik,
                num_samples=args.evaluate_num_samples,
                batch_size=args.evaluate_batch_size,
                normalise=not args.unnormalised,
            ),
        )
    ]

    # See if the point was to just load everything.
    if args.load:
        return {
            "wd": wd,
            "gen_train": gen_train,
            "gen_cv": gen_cv,
            "gens_eval": gens_eval,
            "model": model,
        }

    # The user can just want to see some statistics about the model.
    if args.no_action:
        exit()

    if args.evaluate:
        # Perform evaluation.
        if args.evaluate_last:
            name = "model-last.torch"
        else:
            name = "model-best.torch"
        model.load_state_dict(
            patch_model(torch.load(wd.file(name), map_location=device))["weights"]
        )

        num_samples = 1000
        wd = WorkingDirectory(
            *args.root,
            *(args.subdir or ()),
            data_dir,
            *((f"x{args.dim_x}_y{args.dim_y}",) if hasattr(args, "dim_x") else ()),
            "convcnp",
            *((args.arch,) if hasattr(args, "arch") else ()),
            args.objective,
            str(args.epochs),
            f"eval_{num_samples}",
            log=f"log{suffix}.txt",
            diff=f"diff{suffix}.txt",
            observe=observe,
        )

        # Evaluate different context sets
        import json
        gen = gen_cv()
        gen.batch_size = 1
        num_datasets = 100
        logliks = []
        json_data = {}
        datasets = []
        for j in range(num_datasets):
            batch = gen.generate_batch()
            datasets.append(batch)
            state, loglik = generate_AR_prediction(state, model, batch, num_samples=num_samples, path=wd.file(f"noised_AR_pred-{j + 1:03d}.pdf"), config=config)
            logliks.append(loglik)
            json_data[j] = (loglik.item(), batch["contexts"][0][0].numel())
            with open(wd.file("logliks.json"), "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)

        logliks = B.concat(*logliks)
        print(logliks)
        out.kv("Loglik (E)", exp.with_err(logliks, and_lower=True))

    else:
        # Perform training. First, check if we want to resume training.
        start = 0
        if args.resume_at_epoch:
            start = args.resume_at_epoch - 1
            d_last = patch_model(
                torch.load(wd.file("model-last.torch"), map_location=device)
            )
            d_best = patch_model(
                torch.load(wd.file("model-best.torch"), map_location=device)
            )
            model.load_state_dict(d_last["weights"])
            best_eval_lik = d_best["objective"]
        else:
            best_eval_lik = -np.inf

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
                        wd.file(f"model-epoch-{i+1}.torch"),
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
                )

                # The epoch is done. Now evaluate.
                state, val = eval(state, model, objective_cv, gen_cv())

                # Save current model.
                torch.save(
                    {
                        "weights": model.state_dict(),
                        "objective": val,
                        "epoch": i + 1,
                    },
                    wd.file(f"model-last.torch"),
                )

                # Check if the model is the new best. If so, save it.
                if val > best_eval_lik:
                    out.out("New best model!")
                    best_eval_lik = val
                    torch.save(
                        {
                            "weights": model.state_dict(),
                            "objective": val,
                            "epoch": i + 1,
                        },
                        wd.file(f"model-best.torch"),
                    )

                # Visualise a few predictions by the model.
                gen = gen_cv()
                for j in range(5):
                    exp.visualise_noised_1d(
                        model,
                        gen,
                        path=wd.file(f"train-epoch-{i + 1:03d}-{j + 1}.pdf"),
                        config=config,
                    )


if __name__ == "__main__":
    main(data="noised_sawtooth_diff_targ", dim_y=3, epochs=100 , objective="loglik", evaluate=True)
