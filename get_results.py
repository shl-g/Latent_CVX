#
# Copyright (C) 2024, University of Liege, KAUST and University of Oxford
# TELIM research group, http://www.telecom.ulg.ac.be/
# IVUL research group, https://ivul.kaust.edu.sa/
# VGG research group, https://www.robots.ox.ac.uk/~vgg/
# All rights reserved.
# The code is under the LICENSE.md file.
#
# For inquiries contact jan.held@uliege.be
#

import os
import json
import argparse
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    parser.add_argument("--dataset")
    args = parser.parse_args()

    mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
    mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
    tanks_and_temples_scenes = ["truck", "train"]
    deep_blending_scenes = ["drjohnson", "playroom"]
    all_scenes = []
    if args.dataset == "m360":
        all_scenes.extend(mipnerf360_outdoor_scenes)
        all_scenes.extend(mipnerf360_indoor_scenes)
    elif args.dataset == "tandt":
        all_scenes.extend(tanks_and_temples_scenes)
    elif args.dataset == "db":
        all_scenes.extend(deep_blending_scenes)

    print(all_scenes)

    ssim = []
    psnr = []
    lpips = []

    for scene in all_scenes:
        print(scene)
        data = json.load(open(os.path.join(args.path, scene, "results.json")))
        ssim.append(data["ours_30000"]["SSIM"])
        psnr.append(data["ours_30000"]["PSNR"])
        lpips.append(data["ours_30000"]["LPIPS"])

    print(f"ssim: {np.mean(ssim)}")
    print(f"psnr: {np.mean(psnr)}")
    print(f"lpips: {np.mean(lpips)}")