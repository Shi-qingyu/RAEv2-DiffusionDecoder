import math
import sys
import os
import shutil

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision.utils import make_grid, save_image

import util.misc as misc
import util.lr_sched as lr_sched
import torch_fidelity
import copy
import random


RAEJIT_GENERATION_SAMPLE_MODES = {
    "eval_shifted",
    "dino_first_shifted_aligned",
    "shifted_independent_uniform",
    "dino_first_cascaded",
    "dino_first_cascaded_noised",
}


def concat_all_gather(tensor, gather_dim=0) -> torch.Tensor:
    if torch.distributed.get_world_size() == 1:
        return tensor
    output = torch.distributed.nn.functional.all_gather(tensor)
    return torch.cat(output, dim=gather_dim)


def _denormalize_image_batch(images):
    return (images * 0.5 + 0.5).clamp(0.0, 1.0)


def _make_reconstruction_grid(original_images, reconstructed_images):
    paired_images = torch.stack([original_images, reconstructed_images], dim=1).flatten(0, 1)
    return make_grid(paired_images.detach().cpu(), nrow=2, padding=2)


@torch.no_grad()
def visualize_raejit_epoch(model_without_ddp, args, epoch, val_loader, device, log_writer=None):
    if not misc.is_main_process():
        if misc.is_dist_avail_and_initialized():
            torch.distributed.barrier()
        return

    model_was_training = model_without_ddp.training
    model_without_ddp.eval()

    try:
        images, labels = next(iter(val_loader))
        num_vis = min(args.vis_num, images.size(0))
        images = images[:num_vis].to(device, non_blocking=True).to(torch.float32).div_(255)
        images = images * 2.0 - 1.0
        labels = labels[:num_vis].to(device, non_blocking=True)

        rng_devices = [torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=rng_devices):
            torch.manual_seed(args.seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(args.seed)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == "cuda"):
                reconstructed_images = model_without_ddp.reconstruction(images, labels)

                generated_images = None
                if args.sample_mode in RAEJIT_GENERATION_SAMPLE_MODES:
                    generated_labels = torch.linspace(
                        0, args.class_num - 1, steps=num_vis, device=device
                    ).long()
                    if args.label_drop_prob == 1.0:
                        generated_labels = torch.full_like(generated_labels, args.class_num)
                    generated_images = model_without_ddp.generate(generated_labels)

        original_images = _denormalize_image_batch(images)
        reconstructed_images = _denormalize_image_batch(reconstructed_images)
        reconstruction_grid = _make_reconstruction_grid(original_images, reconstructed_images)

        save_folder = os.path.join(args.output_dir, "visualizations")
        os.makedirs(save_folder, exist_ok=True)
        save_image(
            reconstruction_grid,
            os.path.join(save_folder, f"epoch_{epoch:04d}_reconstruction.png"),
        )
        if log_writer is not None:
            log_writer.add_image("visual/reconstruction", reconstruction_grid, epoch)

        if generated_images is not None:
            generated_images = _denormalize_image_batch(generated_images)
            generation_grid = make_grid(generated_images.detach().cpu(), nrow=min(num_vis, 4), padding=2)
            save_image(
                generation_grid,
                os.path.join(save_folder, f"epoch_{epoch:04d}_generation.png"),
            )
            if log_writer is not None:
                log_writer.add_image("visual/generation", generation_grid, epoch)
    finally:
        if model_was_training:
            model_without_ddp.train()
        if misc.is_dist_avail_and_initialized():
            torch.distributed.barrier()

def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (x, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # normalize image to [-1, 1]
        x = x.to(device, non_blocking=True).to(torch.float32).div_(255)
        x = x * 2.0 - 1.0
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss, results_dict = model(x, labels)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        # print(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf')).detach().cpu().item())
        optimizer.step()

        torch.cuda.synchronize()

        model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value)
        metric_logger.update(**results_dict)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        reduced_results = {}
        for k, v in sorted(results_dict.items()):
            reduced_results[k] = misc.all_reduce_mean(v)

        if log_writer is not None:
            # Use epoch_1000x as the x-axis in TensorBoard to calibrate curves.
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)
                for k, v in reduced_results.items():
                    log_writer.add_scalar(k, v, epoch_1000x)


def evaluate(model_without_ddp, args, epoch, batch_size=64, log_writer=None):

    model_without_ddp.eval()
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    num_images = 1000 if epoch == 0 else args.num_images 
    num_steps = num_images // (batch_size * world_size) + 1

    # Construct the folder name for saving generated images.
    save_folder = os.path.join(
        args.output_dir,
        "gen-{}-steps{}-cfg{}-interval{}-{}-image{}-res{}".format(
            model_without_ddp.method, model_without_ddp.steps, model_without_ddp.cfg_scale,
            model_without_ddp.cfg_interval[0], model_without_ddp.cfg_interval[1], num_images, args.img_size
        )
    )
    print("Save to:", save_folder)
    if misc.get_rank() == 0 and not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # switch to ema params, hard-coded to be the first one
    model_state_dict = None
    if args.generation_ema != 'none':
        model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
            assert name in ema_state_dict
            maybe_ema_state_dict = {
                '1': model_without_ddp.ema_params1,
                '2': model_without_ddp.ema_params2,
            }[args.generation_ema]
            ema_state_dict[name] = maybe_ema_state_dict[i]
        print("Switch to ema")
        model_without_ddp.load_state_dict(ema_state_dict)

    # ensure that the number of images per class is equal.
    class_num = args.class_num
    assert num_images % class_num == 0, "Number of images per class must be the same"
    class_label_gen_world = np.arange(0, class_num).repeat(num_images // class_num)
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])

    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))

        start_idx = world_size * batch_size * i + local_rank * batch_size
        end_idx = start_idx + batch_size
        labels_gen = class_label_gen_world[start_idx:end_idx]
        labels_gen = torch.Tensor(labels_gen).long().cuda()
        if args.label_drop_prob == 1.0:
            labels_gen = labels_gen * 0 + 1000

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            sampled_images = model_without_ddp.generate(labels_gen)

        torch.distributed.barrier()

        # denormalize images
        sampled_images = (sampled_images + 1) / 2
        sampled_images = sampled_images.detach().cpu()

        # distributed save images
        for b_id in range(sampled_images.size(0)):
            img_id = i * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + b_id
            if img_id >= num_images:
                break
            gen_img = np.round(np.clip(sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_folder, '{}.png'.format(str(img_id).zfill(5))), gen_img)

    torch.distributed.barrier()

    # back to no ema
    if model_state_dict is not None:
        print("Switch back from ema")
        model_without_ddp.load_state_dict(model_state_dict)

    # compute FID and IS
    if log_writer is not None:
        if args.img_size == 256:
            fid_statistics_file = 'fid_stats/jit_in256_stats.npz'
        elif args.img_size == 512:
            fid_statistics_file = 'fid_stats/jit_in512_stats.npz'
        else:
            raise NotImplementedError
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=save_folder,
            input2=None,
            fid_statistics_file=fid_statistics_file,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )
        fid = metrics_dict['frechet_inception_distance']
        inception_score = metrics_dict['inception_score_mean']
        postfix = "_cfg{}_res{}".format(model_without_ddp.cfg_scale, args.img_size)
        log_writer.add_scalar('fid{}'.format(postfix), fid, epoch)
        log_writer.add_scalar('is{}'.format(postfix), inception_score, epoch)
        print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))


        if not args.keep_images:
            shutil.rmtree(save_folder)

    torch.distributed.barrier()


@torch.inference_mode()
def compute_psnr_torch_batch(original, recon, data_range: float = 1.0):
    """computes psnr for a batch of images using pytorch operations."""
    mse_per_sample = F.mse_loss(original, recon, reduction="none").mean(dim=[1, 2, 3])
    psnr_per_sample = 10.0 * torch.log10(data_range**2 / mse_per_sample)
    return psnr_per_sample


def evaluate_reconstruction(model_without_ddp, args, epoch, val_loader, device, log_writer=None):

    model_without_ddp.eval()
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    num_images = 1000 if epoch == 0 else args.num_images 

    # Construct the folder name for saving generated images.
    save_folder = os.path.join(
        args.output_dir,
        "rec-{}-steps{}-cfg{}-interval{}-{}-image{}-res{}".format(
            model_without_ddp.method, model_without_ddp.steps, model_without_ddp.cfg_scale,
            model_without_ddp.cfg_interval[0], model_without_ddp.cfg_interval[1], num_images, args.img_size
        )
    )
    print("Save to:", save_folder)
    if misc.get_rank() == 0 and not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # switch to ema params, hard-coded to be the first one
    model_state_dict = None
    if args.generation_ema != 'none':
        model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
            assert name in ema_state_dict
            maybe_ema_state_dict = {
                '1': model_without_ddp.ema_params1,
                '2': model_without_ddp.ema_params2,
            }[args.generation_ema]
            ema_state_dict[name] = maybe_ema_state_dict[i]
        print("Switch to ema")
        model_without_ddp.load_state_dict(ema_state_dict)

    psnr_values_local = []

    for i, batch in enumerate(val_loader):
        print("Generation step {}/{}".format(i, len(val_loader)))

        x, labels = batch
        x = x.to(device, non_blocking=True).to(torch.float32).div_(255)
        x = x * 2.0 - 1.0
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            reconstructed_images = model_without_ddp.reconstruction(x, labels)

        torch.distributed.barrier()

        # denormalize images
        reconstructed_images = (reconstructed_images + 1) / 2
        cur_psnr = compute_psnr_torch_batch(x * 0.5 + 0.5, reconstructed_images, data_range=1.0)
        psnr_values_local.extend(cur_psnr.cpu().tolist())
        reconstructed_images = reconstructed_images.detach().cpu()


        # distributed save images
        for b_id in range(reconstructed_images.size(0)):
            img_id = i * reconstructed_images.size(0) * world_size + local_rank * reconstructed_images.size(0) + b_id
            if img_id >= num_images:
                break
            gen_img = np.round(np.clip(reconstructed_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_folder, '{}.png'.format(str(img_id).zfill(5))), gen_img)

    torch.distributed.barrier()

    # back to no ema
    if model_state_dict is not None:
        print("Switch back from ema")
        model_without_ddp.load_state_dict(model_state_dict)

    psnr_values_local_tensor = torch.tensor(psnr_values_local, device=device, dtype=torch.float32)
    psnr_gathered_tensor = concat_all_gather(psnr_values_local_tensor, gather_dim=0)
    
    if misc.is_main_process():
        # psnr_gathered_tensor now contains the concatenated PSNR values from all ranks
        mean_psnr = psnr_gathered_tensor.mean().item()
        print(f"Average PSNR (all ranks): {mean_psnr:.4f}")
    else:
        mean_psnr = 0.0

    # compute FID and IS
    if log_writer is not None:
        if args.img_size == 256:
            fid_statistics_file = 'fid_stats/val_fid_statistics_file_256.npz'
        else:
            raise NotImplementedError
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=save_folder,
            input2=None,
            fid_statistics_file=fid_statistics_file,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )
        fid = metrics_dict['frechet_inception_distance']
        inception_score = metrics_dict['inception_score_mean']
        postfix = "_cfg{}_res{}".format(model_without_ddp.cfg_scale, args.img_size)
        log_writer.add_scalar('fid{}'.format(postfix), fid, epoch)
        log_writer.add_scalar('is{}'.format(postfix), inception_score, epoch)
        print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))
        if not args.keep_images:
            shutil.rmtree(save_folder)

    torch.distributed.barrier()
