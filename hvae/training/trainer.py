import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from torch import distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from ..utils.datasets import SpecDataset, make_dataloader
from ..utils.logging_wrapper import LoggerWrapper
from .model import WaveNet


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def get_mem_usage() -> str:
    if not torch.cuda.is_available():
        return ""
    used_gb = torch.cuda.max_memory_allocated() / (2**30)
    total_gb = torch.cuda.get_device_properties(0).total_memory / (2**30)
    torch.cuda.reset_peak_memory_stats()
    return "Max mem. usage: {:.2f}/{:.2f}GiB".format(used_gb, total_gb)


def initialize_for_training(
    rank: int,
    world_size: int,
    config: dict,
    data_path: Path,
    save_directory: Path,
) -> tuple[
    WaveNet,
    optim.Optimizer,
    optim.lr_scheduler._LRScheduler,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    int,
    LoggerWrapper,
]:
    """Initializes all the necessary components for training. If the save directory already exists,
    it will attempt to restart at the last checkpoint."""
    is_master = rank == 0
    save_directory.mkdir(exist_ok=True, parents=True)
    log_path = save_directory / "train_log.log"
    if not torch.cuda.is_available():
        raise RuntimeError("Cuda not available!")

    is_resuming = bool(next(save_directory.glob("weights_*.pt"), False))
    weight_sort_key = lambda x: int(x.stem.split("_")[-1])
    recent_weights_path = max(
        save_directory.glob("weights_*.pt"), key=weight_sort_key, default=None
    )

    # Log to stdout and file
    logger = LoggerWrapper(rank, log_path)

    if world_size > 1:
        # setup ddp:
        logger.info(f"Initiating ddp with rank {rank} and world size {world_size}")
        setup(rank, world_size)

    # Save the config used for training
    if is_master:
        with open(save_directory / "config.json", "w") as ctx:
            json.dump(config, ctx, indent=4)

    """    test_model = WaveNet(
        input_dim=80,
        num_layers_per_block=2,
        kernel_size=[7, 7, 7, 7],
        num_channels=[64, 64, 64, 64],
        dilation=[3, 3, 2, 2],
        downsample_factors=[16, 16, 16, 16],
        latent_dim=[16, 16, 16, 16],
    )"""
    model = WaveNet(**config["model"])

    if world_size > 1:
        model = model.to(rank)
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        # model.compile()
    else:
        model = model.cuda()
        # model.compile()

    # Ensure initial weights are synced across processes
    if world_size > 1:
        initial_weights_path = (
            recent_weights_path
            if is_resuming
            else save_directory / "initial_weights.pt"
        )

        if is_master and not is_resuming:
            initial_weights = model.state_dict()
            torch.save(initial_weights, initial_weights_path)

        dist.barrier()
        initial_weights = torch.load(
            initial_weights_path,
            map_location={"cuda:0": f"cuda:{rank}"},
            weights_only=True,
        )
        print(f"Rank {rank}: loading weights from {initial_weights_path}")
        model.load_state_dict(initial_weights)
        dist.barrier()

        if is_master and not is_resuming:
            os.remove(initial_weights_path)

        torch.manual_seed(rank)
    else:
        if is_resuming:
            model.load_state_dict(torch.load(recent_weights_path, weights_only=True))
            logger.info(f"Loaded weights from {recent_weights_path}")

    train_dataloader = make_dataloader(data_path, config=config)
    train_dset: SpecDataset = train_dataloader.dataset
    train_index_path = save_directory / "train_dates.txt"
    val_dataloader = make_dataloader(data_path, config=config)
    val_dset: SpecDataset = val_dataloader.dataset
    val_index_path = save_directory / "val_dates.txt"

    if is_master and not is_resuming:
        # Manually generate a consistent test/train split for all processes
        with open(train_index_path, "w") as f:
            f.write("\n".join(train_dset.train_keys))
        with open(val_index_path, "w") as f:
            f.write("\n".join(train_dset.val_keys))

    if world_size > 1:
        dist.barrier()  # Ensure index exists before reloading indices

    with open(train_index_path, "r") as ctx:
        train_keys = ctx.read().splitlines()
    with open(val_index_path, "r") as ctx:
        val_keys = ctx.read().splitlines()

    train_dset.dates_for_training = train_keys
    val_dset.dates_for_training = val_keys

    logger.info(repr(model))

    opt_config = config["optimization"]

    opt = optim.SGD(
        model.parameters(),
        lr=opt_config["initial_learning_rate"],
        momentum=opt_config["momentum"],
    )

    n_warmup_steps = opt_config["num_warmup_steps"]
    warmup_sched = optim.lr_scheduler.LinearLR(opt, 0.1, 1.0, n_warmup_steps)
    cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=opt_config["num_decay_steps"],
        eta_min=opt_config["min_learning_rate"],
        verbose=False,
    )
    sched = optim.lr_scheduler.SequentialLR(
        opt, [warmup_sched, cosine_sched], milestones=[n_warmup_steps]
    )

    epoch = 0
    if is_resuming:
        epoch = int(recent_weights_path.stem.split("_")[-1])
        step_num = epoch * opt_config["num_updates_per_ckpt"]
        for _ in range(step_num):
            sched.step()

    return model, opt, sched, train_dataloader, val_dataloader, epoch, logger


def forward_pass(
    model: WaveNet,
    batch: torch.Tensor,
    rank: int,
    reduce_loss: bool = True,
    loss_type: Literal["MSE", "IS"] = "MSE",
) -> torch.Tensor:
    audio = batch.to(rank)

    # encoding
    # posterior_params = model.encode(audio)
    # reconstruction = model.decode(posterior_params, num_timescales=1)
    posterior_params, reconstruction = model(audio)

    # Under a gaussian model of the likelihood p(x|z), the loss is a combination of MSE and KL divergence
    if loss_type == "MSE":
        recon_loss = (
            torch.square(reconstruction - audio).flatten(start_dim=1).mean(dim=1)
        )
    else:
        # IS loss rewritten to use log power spectrum
        diff = audio - reconstruction
        recon_loss = (torch.exp(diff) - diff - 1).flatten(start_dim=1).mean(dim=1)
    # Assuming a prior p(z) = N(0, I):
    means, log_variances = posterior_params[
        0
    ]  # each have shape (batch, num_latents, time)

    # Using eq from https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
    first_term = log_variances.sum(dim=-2)  # log (det Sigma_2 / det Sigma_1)
    sigma_2_inv = torch.exp(-log_variances)  # diag elements of matrix Sigma_2^-1
    second_term = sigma_2_inv.sum(dim=-2)  # trace [Sigma_2^-1 Sigma_1]
    third_term = (means * means * sigma_2_inv).sum(
        dim=-2
    )  # (mu_2 - mu_1)^T Sigma_2^-1 (mu_2 - mu_1), mu_1=0
    kl_divergence = 0.5 * (first_term + second_term + third_term - means.shape[-2])
    kl_divergence = kl_divergence.mean(dim=-1)  # mean over time, shape is now (batch,)

    loss = recon_loss + kl_divergence
    if not reduce_loss:
        return recon_loss, kl_divergence

    return loss.mean(), recon_loss, kl_divergence


def train(
    rank: int,
    world_size: int,
    config: dict,
    data_path: Path,
    save_directory: Optional[Path] = None,
):
    is_master = rank == 0
    log_interval = config["optimization"]["num_updates_per_log"]

    model, opt, sched, train_dloader, val_dloader, epoch, logger = (
        initialize_for_training(rank, world_size, config, data_path, save_directory)
    )

    # Logs the time at the end of each minibatch, used to estimate speed
    proc_times = deque(maxlen=30)

    # Logs the number of minibatches containing NaN or inf gradients
    num_failed_batches = 0

    total_num_updates = config["optimization"]["num_weight_updates"]
    mb_per_epoch = config["optimization"]["num_updates_per_ckpt"]
    n_mb_processed_cur_epoch = 0
    n_mb_procd = lambda: epoch * mb_per_epoch + n_mb_processed_cur_epoch

    train_iter = iter(train_dloader)
    val_iter = iter(val_dloader)

    while n_mb_procd() < total_num_updates:
        # Training period
        model.train()
        while n_mb_processed_cur_epoch < mb_per_epoch:
            batch = next(train_iter)  # infinite iterator

            opt.zero_grad()
            loss, recon, kl = forward_pass(
                model,
                batch,
                rank,
                reduce_loss=True,
                loss_type=config["optimization"]["loss_fn"],
            )
            loss.backward()

            # Did a batch fail?
            are_gradients_nan = any(
                (
                    torch.isnan(p.grad).any().cpu().item()
                    if p.grad is not None
                    else False
                )
                for p in model.parameters()
            )
            if (world_size == 1 and not model._clip_gradients()) or are_gradients_nan:
                num_failed_batches += 1
                logger.warn(
                    f"Encountered non-finite gradients at step {n_mb_procd()}. Skipping batch."
                )
                logger.warn(f"Total failed batches: {num_failed_batches}")
                opt.zero_grad()
                sched.step()
                continue

            opt.step()
            sched.step()

            proc_times.append(time.time())
            n_mb_processed_cur_epoch += 1
            # Log progress
            if (n_mb_procd() % log_interval) == 0:
                logger.info(get_mem_usage())
                processing_speed = len(proc_times) / (proc_times[-1] - proc_times[0])
                logger.info(
                    f"Progress: {n_mb_procd()} / {total_num_updates} weight updates."
                )
                loss = loss.detach().cpu().item()
                logger.info(f"Most recent loss: {loss:.3e}")
                recon = recon.detach().cpu().mean().item()
                kl = kl.detach().cpu().mean().item()
                logger.info(f"KL: {kl:.3e},\tRecon: {recon:.3e}")
                logger.info(
                    f"Speed: {processing_speed*len(batch) * world_size:.1f} vocalizations per second"
                )
                eta = (total_num_updates - n_mb_procd()) / processing_speed
                eta_hours = int(eta // 3600)
                eta_minutes = int(eta // 60) - 60 * eta_hours
                eta_seconds = int(eta % 60)
                if not eta_hours:
                    logger.info(
                        f"Est time until end of training: {eta_minutes}:{eta_seconds:0>2d}"
                    )
                else:
                    logger.info(
                        f"Est time until end of training: {eta_hours}:{eta_minutes:0>2d}:{eta_seconds:0>2d}"
                    )
                logger.info(f"Current learning rate: {sched.get_last_lr()[0]:.3e}")

                logger.info("")

        # Validation period
        epoch += 1
        n_mb_processed_cur_epoch = 0

        model.eval()

        losses = []
        with torch.no_grad():
            for _ in tqdm(range(200)):
                batch = next(val_iter)  # this one is also infinite...
                losses.append(
                    forward_pass(model, batch, rank, reduce_loss=False)[0].cpu().numpy()
                )
        losses = np.concatenate(losses)

        logger.info(f"Validation mean loss: {losses.mean():.3f}")

        logger.info("Saving state")
        if is_master:
            wpath = save_directory / f"weights_{epoch}.pt"
            torch.save(model.state_dict(), wpath)
        logger.info("")
    logger.info("Done.")

    if world_size > 1:
        cleanup()


def test(
    config: dict,
    data_path: Path,
    save_directory: Path,
):
    model, _, _, _, test_dloader, _, logger = initialize_for_training(
        0, 1, config, data_path, save_directory
    )

    model.eval()

    orig = []
    recon = []
    posterior_params = []

    test_iter = iter(test_dloader)
    with torch.no_grad():
        for _ in tqdm(range(1)):
            batch = next(test_iter)
            audio = batch.cuda()
            posterior, reconstruction = model(audio)
            orig.append(audio.cpu().numpy())
            recon.append(reconstruction.cpu().numpy())
            posterior_params.append(posterior)

    orig = np.concatenate(orig).astype(np.float16)
    np.save(save_directory / "orig.npy", orig)
    recon = np.concatenate(recon).astype(np.float16)
    np.save(save_directory / "recon.npy", recon)

    means = {f"layer_{i}": [] for i in range(len(posterior_params[0]))}
    log_vars = {f"layer_{i}": [] for i in range(len(posterior_params[0]))}

    for batch in posterior_params:
        for i, (mean, log_var) in enumerate(batch):
            means[f"layer_{i}"].append(mean.cpu().numpy().astype(np.float16))
            log_vars[f"layer_{i}"].append(log_var.cpu().numpy().astype(np.float16))

    for i in range(len(posterior_params[0])):
        np.save(
            save_directory / f"means_layer_{i}.npy", np.concatenate(means[f"layer_{i}"])
        )
        np.save(
            save_directory / f"log_vars_layer_{i}.npy",
            np.concatenate(log_vars[f"layer_{i}"]),
        )
