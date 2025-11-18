#
# The original code is under the following copyright:
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE_GS.md file.
#
# For inquiries contact george.drettakis@inria.fr
#
# The modifications of the code are under the following copyright:
# Copyright (C) 2024, University of Liege, KAUST and University of Oxford
# TELIM research group, http://www.telecom.ulg.ac.be/
# IVUL research group, https://ivul.kaust.edu.sa/
# VGG research group, https://www.robots.ox.ac.uk/~vgg/
# All rights reserved.
# The modifications are under the LICENSE.md file.
#
# For inquiries contact jan.held@uliege.be
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from convex_renderer import render
import sys
import PIL
from PIL import Image
import torchvision.transforms as T
from scene import Scene, ConvexModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, adapt
from utils.important_functions_compilation import StableDiffusion, calc_text_embeddings, RGB_to_latent
import numpy as np
import matplotlib.pyplot as plt
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import time
import lpips


def save_image(convexes, viewpoint_cam, pipe, bg, text):
    print(f"[DEBUG] Saving image to: '{text}'")  # <--- ADD THIS
    render_pkg = render(viewpoint_cam, convexes, pipe, bg)
    image, _, _, _ = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
    image_array = image.detach().permute(1, 2, 0).cpu().numpy()
    image_array = np.clip(image_array, 0, 1)
    plt.imsave(text, image_array)
    return image_array

def training(
        dataset,   
        opt, 
        pipe,
        light, 
        outdoor,
        testing_iterations,
        save_iterations,
        checkpoint, 
        debug_from,
        ref_text,
        pcd_points,
        num_cams,
        prune_factor
        ):
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    # Load parameters, convexes and scene
    #opt = adapt(opt, light, outdoor)

    convexes = ConvexModel(dataset.sh_degree, opt.feature_ratio)
    scene = Scene(dataset, convexes, opt.set_opacity, opt.convex_size, opt.nb_points, opt.set_delta, opt.set_sigma, opt.densify_grad_threshold, light, pcd_points=pcd_points, num_cameras=num_cams)
    convexes.training_setup(opt, opt.lr_mask, opt.feature_lr, opt.opacity_lr, opt.lr_delta, opt.lr_sigma, opt.lr_convex_points_init, opt.lr_convex_points_final, opt.shifting_cloning, opt.scaling_cloning, opt.sigma_scaling_cloning, opt.delta_scaling_cloning, opt.opacity_cloning)
    Diff_model = StableDiffusion(device="cuda")
    text_z_list = calc_text_embeddings(ref_text, Diff_model)
    debug_save_dir = os.path.join(dataset.model_path, "debug images")

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        convexes.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        convexes.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration > 800 and iteration % 500 == 0:
            convexes.oneupSHdegree()
        if iteration > 2000: # added to reduce the densification interval
            opt.densification_interval = 1001
            
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, convexes, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii, scaling, density_factor, viewspace_sigma = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["scaling"], render_pkg["density_factor"], render_pkg["viewspace_sigma"]

        # Calculate text embeddings
        cam_direction = viewpoint_cam.dir

        text_z = text_z_list[cam_direction]
        
        # Calculate Loss
        #print("Rendered image shape:", image.shape)
        # After rendering
        #print("Image stats:", image.min().item(), image.max().item(), image.mean().item())
        image = image.to(device="cuda")

        #print("latent_image shape:", latent_image.shape)
        
        #print("Image requires_grad:", image.requires_grad)
        loss = Diff_model.train_step(text_z, image)
        loss = loss + 0.0005*torch.mean((torch.sigmoid(convexes._mask)))
        loss = loss + 0.0005 * torch.mean(convexes.max_radii2D)
        loss.backward()

        
        """current_dir = os.path.dirname(os.path.abspath(__file__))
        target_img_path = os.path.join(current_dir, "harlin.jpg")
        # Transform: resize to 512x512 and convert to tensor
        transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor()
        ])
        
        # Load and preprocess the target image
        harlin_image = Image.open(target_img_path).convert("RGB")
        harlin_image = transform(harlin_image).unsqueeze(0).to("cuda") 
        loss = torch.nn.functional.l1_loss(image, harlin_image)
        loss = loss + 0.0005*torch.mean((torch.sigmoid(convexes._mask)))
        loss.backward()"""
        """for i, group in enumerate(convexes.optimizer.param_groups):
            print(f"\nParam group {i}: LR={group['lr']}")
            for p in group['params']:
                print(f"  Shape={p.shape}, requires_grad={p.requires_grad}, grad={None if p.grad is None else p.grad.abs().mean().item()}")"""
        # This part is new - remove later
        iter_end.record()
        
        # #debug images prints -start
        # if iteration % 50 == 0 or iteration < 12:
        #     os.makedirs(debug_save_dir, exist_ok=True)
        #
        #     img_tensor = image.detach().cpu().clamp(0, 1)  # [3, H, W]
        #     img_np = img_tensor.permute(1, 2, 0).numpy()   # [H, W, 3]
        #
        #     # Convert to uint8
        #     img_uint8 = (img_np * 255).astype(np.uint8)
        #
        #     filename = os.path.join(debug_save_dir, f"iter_{iteration:06d}.png")
        #     PIL.Image.fromarray(img_uint8).save(filename)
        #     print(f"[INFO] Saved debug render at: {filename}")
        #
        #     # print params of convexes:
        #     num_total_points = convexes._number_of_points
        #     num_convexes = num_total_points // convexes.nb_points
        #
        #     print(f"Number of convexes: {num_convexes}")
        #
        #     radii = convexes.max_radii2D  # Tensor of shape [num_convexes]
        #     print(f"Mean convex size: {radii.mean().item():.6f}")
        #     print(f"Min convex size: {radii.min().item():.6f}")
        #     print(f"Max convex size: {radii.max().item():.6f}")
        #     # End of If condition
        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            
            #training_report(tb_writer, iteration, None, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if iteration in save_iterations:
                print(f"\n[ITER {iteration}] Saving convexes")
                scene.save(iteration, light)

            pruning_iteration = prune_factor * iteration
            if pruning_iteration < opt.densify_until_iter:
                convexes.max_radii2D[visibility_filter] = torch.max(convexes.max_radii2D[visibility_filter], radii[visibility_filter])
                convexes.add_densification_stats(viewspace_point_tensor, viewspace_sigma, visibility_filter, scaling, density_factor)


                # Remove convexes shape with opacity < threshold
                if pruning_iteration > opt.densify_from_iter and pruning_iteration % opt.densification_interval == 0:
                    size_threshold = opt.remove_size_threshold if pruning_iteration > opt.opacity_reset_interval else None
                    convexes.densify_and_prune(opt.min_opacity, opt.mask_threshold, scene.cameras_extent, size_threshold)

                # Reset the opacity
                if pruning_iteration % opt.opacity_reset_interval  == 0 or (dataset.white_background and pruning_iteration == opt.densify_from_iter):
                    convexes.reset_opacity(opt.opacity_reset)
            else:
                if pruning_iteration % 1000 == 0:
                    convexes.only_prune(opt.min_opacity, opt.mask_threshold)

                if pruning_iteration % opt.opacity_reset_interval == 0 and pruning_iteration < opt.reset_opacity_until:
                    convexes.reset_opacity(opt.opacity_reset)

            # Optimizer step
            if iteration < opt.iterations:
                convexes.optimizer.step()
                convexes.optimizer.zero_grad(set_to_none = True)


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.convexes, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    lpips_test += lpips_fn(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])       
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])  
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))

                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.convexes.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.convexes.get_convex_points.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--feature_lr", type=float, default=0.025, help="Learning of the features")
    parser.add_argument("--feature_ratio", type=float, default=1, help="Ratio between the DC and rest features")
    parser.add_argument("--lr_convex_points_init", type=float, default=0.0005, help="Initial Position LR value")
    op = OptimizationParams(parser)
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[500, 1_000, 1_500, 2_000, 2_500, 3_000, 3_500, 4_000, 4_500, 5_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[500, 1_000, 1_500, 2_000, 2_500, 3_000, 3_500, 4_000, 4_500, 5_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    parser.add_argument("--light", action="store_true", default=False)
    parser.add_argument("--outdoor", action="store_true", default=False)

    parser.add_argument("--prompt", type=str, help="Text prompt for training")
    parser.add_argument("--num_points", type=int, default=25_000, help="PCD number of points")
    parser.add_argument("--num_cams", type=int, default=12, help="Number of Cameras")
    parser.add_argument("--prune_factor", type=int, default=1, help="Prune factor")



    args = parser.parse_args(sys.argv[1:])
    if args.prompt is None:
        parser.error(
            "The --prompt argument is required. Please provide a prompt like: --prompt 'a highly detailed sand castle'")
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    lpips_fn = lpips.LPIPS(net='vgg').to(device="cuda")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args),
             op.extract(args),
             pp.extract(args),
             args.light,
             args.outdoor,
             args.test_iterations,
             args.save_iterations,
             args.start_checkpoint,
             args.debug_from,
             args.prompt,
             args.num_points,
             args.num_cams,
             args.prune_factor
             )
    
    # All done
    print("\nTraining complete.")
