#
# Copyright (C) 2024, Inria, University of Liege, KAUST and University of Oxford
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# TELIM research group, http://www.telecom.ulg.ac.be/
# IVUL research group, https://ivul.kaust.edu.sa/
# VGG research group, https://www.robots.ox.ac.uk/~vgg/
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  jan.held@uliege.be
#


import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from convex_renderer import render
import torchvision
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from convex_renderer import ConvexModel
import numpy as np
from utils.render_utils import generate_path, create_videos

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    args = get_combined_args(parser)
    print("Creating video for " + args.model_path)

    dataset, pipe = model.extract(args), pipeline.extract(args)

    convexes = ConvexModel(dataset.sh_degree)
    scene = Scene(dataset, convexes, None, None, None, None, None, None, light=False, load_iteration=args.iteration, shuffle=False)

    #bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    bg_color = [1,1,1] # forcing background to be white
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    traj_dir = os.path.join(args.model_path, 'traj')
    os.makedirs(traj_dir, exist_ok=True)

    render_path = os.path.join(traj_dir, "renders")
    os.makedirs(render_path, exist_ok=True)
    
    n_frames = 240*5
    cam_traj = generate_path(scene.getTrainCameras(), n_frames=n_frames)

    with torch.no_grad():
        for idx, view in enumerate(tqdm(cam_traj, desc="Rendering progress")):
            rendering = render(view, convexes, pipe, background)["render"]
            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(rendering, os.path.join(traj_dir, "renders", '{0:05d}'.format(idx) + ".png"))

    create_videos(base_dir=traj_dir,
                input_dir=traj_dir, 
                out_name='render_traj', 
                num_frames=n_frames)
