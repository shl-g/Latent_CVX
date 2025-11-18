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

import torch
import math
from diff_convex_rasterization import ConvexRasterizationSettings, ConvexRasterizer
from scene.convex_model import ConvexModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : ConvexModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_convex_points[:,0,:].squeeze(), dtype=pc.get_convex_points.dtype, requires_grad=True, device="cuda") + 0
    scaling = torch.zeros_like(pc.get_convex_points[:,0,0].squeeze(), dtype=pc.get_convex_points.dtype, requires_grad=True, device="cuda").detach()
    density_factor = torch.zeros_like(pc.get_convex_points[:,0,0].squeeze(), dtype=pc.get_convex_points.dtype, requires_grad=True, device="cuda").detach()

    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = ConvexRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = ConvexRasterizer(raster_settings=raster_settings)

    opacity = pc.get_opacity
    convex_points = pc.get_convex_points_flatten
    delta = pc.get_delta
    sigma = pc.get_sigma
    new_sigma = sigma.clone().requires_grad_(True)
    new_sigma.retain_grad()
    new_delta = delta.clone().requires_grad_(True)
    new_delta.retain_grad()
    num_points_per_convex = pc.get_num_points_per_convex
    cumsum_of_points_per_convex = pc.get_cumsum_of_points_per_convex
    number_of_points = pc.get_number_of_points
    means2D = screenspace_points

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features

    else:
        colors_precomp = override_color



    mask = ((torch.sigmoid(pc._mask) > 0.01).float()- torch.sigmoid(pc._mask)).detach() + torch.sigmoid(pc._mask)
    opacity = opacity * mask

    # Rasterize visible Convexes to image, obtain their radii (on screen). 
    rendered_image, radii, scaling, density_factor, allmap  = rasterizer(
        convex_points=convex_points,
        delta=new_delta,
        sigma=new_sigma,
        num_points_per_convex = num_points_per_convex,
        cumsum_of_points_per_convex = cumsum_of_points_per_convex,
        number_of_points = number_of_points,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        means2D = means2D,
        scaling = scaling,
        density_factor = density_factor
       )


    rets =  {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii, 
            "scaling": scaling,
            "density_factor": density_factor,
            "viewspace_sigma": new_sigma,
            "viewspace_delta": new_delta
            }


    return rets





 