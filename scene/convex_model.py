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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from torch import nn
import os
from utils.system_utils import mkdir_p
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud


def fibonacci_sphere(x, y, z, radii, nb_points):
    # Prepare storage for the generated points
    points_per_convex = torch.zeros((x.shape[0], nb_points, 3), device=x.device)

    # Generate nb_points on a unit sphere using the Fibonacci lattice
    for i in range(nb_points):
        # z-coordinates, linearly spaced between 1 and -1, converted to tensor
        z_coord = torch.tensor(1 - (2 * i / (nb_points - 1)), device=x.device)  # Tensor
        
        # Calculate the radial distance in the xy-plane
        radii_xy = torch.sqrt(1 - z_coord**2)  # Tensor, radial distance in the xy-plane

        # Theta, spaced by the golden angle
        theta = torch.pi * (3.0 - torch.sqrt(torch.tensor(5.0))) * i  # Scalar, but used later in tensor ops

        # Generate unit vectors for each point on the sphere
        x_unit = radii_xy * torch.cos(torch.tensor(theta, device=x.device))  # Tensor
        y_unit = radii_xy * torch.sin(torch.tensor(theta, device=x.device))  # Tensor
        z_unit = z_coord  # Already a tensor

        # Stack the unit vector (shape: [3]) and scale it by radii (shape: [100, 1])
        unit_sphere_point = torch.stack([x_unit, y_unit, z_unit], dim=0)  # Shape: [3]
        
        # Apply the scaling by radii and add the center coordinates
        points_per_convex[:, i, :] = radii * unit_sphere_point + torch.stack([x, y, z], dim=1)

    return points_per_convex


class ConvexModel:

    def setup_functions(self):
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.exponential_activation = torch.exp
        self.inverse_exponential_activation = torch.log

    def __init__(self, sh_degree : int, feature_ratio : float = 1):
        self._convex_points = torch.empty(0)
        self._delta = torch.empty(0)
        self._sigma = torch.empty(0)
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.max_density_factor = torch.empty(0)
        self.convex_accumulation_gradient_accum = torch.empty(0)
        self.sigma_accumulation_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.spatial_lr_scale = 0
        self.feature_ratio = feature_ratio

        self._mask = torch.empty(0)

        self.max_scaling = torch.empty(0)

        self._num_points_per_convex = torch.empty(0)
        self._cumsum_of_points_per_convex = torch.empty(0)
        self._number_of_points = 0

        self.nb_points = 0
        self.shifting_cloning = 0
        self.scaling_cloning = 0
        self.sigma_scaling_cloning = 0
        self.delta_scaling_cloning = 0
        self.opacity_cloning = 0

        self.max_grad = 0
        self.setup_functions()

    def save(self, path, light):

        mkdir_p(path)

        point_cloud_state_dict = {}


        if light:
            point_cloud_state_dict["convex_points"] = self._convex_points.to(torch.float16)
            point_cloud_state_dict["delta"] = self._delta.to(torch.float16)
            point_cloud_state_dict["sigma"] = self._sigma.to(torch.float16)
            point_cloud_state_dict["active_sh_degree"] = self.active_sh_degree  # Assuming this is not a tensor
            point_cloud_state_dict["features_dc"] = self._features_dc.to(torch.float16)
            point_cloud_state_dict["features_rest"] = self._features_rest.to(torch.float16)
            point_cloud_state_dict["opacity"] = self._opacity.to(torch.float16)
        else:
            point_cloud_state_dict["convex_points"] = self._convex_points
            point_cloud_state_dict["delta"] = self._delta
            point_cloud_state_dict["sigma"] = self._sigma
            point_cloud_state_dict["active_sh_degree"] = self.active_sh_degree
            point_cloud_state_dict["features_dc"] = self._features_dc
            point_cloud_state_dict["features_rest"] = self._features_rest
            point_cloud_state_dict["opacity"] = self._opacity

        torch.save(point_cloud_state_dict, os.path.join(path, 'point_cloud_state_dict.pt'))

        hyperparameters = {}

        hyperparameters["max_radii2D"] = self.max_radii2D
        hyperparameters["convex_accumulation_gradient_accum"] = self.convex_accumulation_gradient_accum
        hyperparameters["denom"] = self.denom
        hyperparameters["spatial_lr_scale"] = self.spatial_lr_scale
        hyperparameters["num_points_per_convex"] = self._num_points_per_convex
        hyperparameters["cumsum_of_points_per_convex"] = self._cumsum_of_points_per_convex
        hyperparameters["number_of_points"] = self._number_of_points
        hyperparameters["max_scaling"] = self.max_scaling
        hyperparameters["max_density_factor"] = self.max_density_factor
        hyperparameters["sigma_accumulation_gradient_accum"] = self.sigma_accumulation_gradient_accum

        torch.save(hyperparameters, os.path.join(path, 'hyperparameters.pt'))

    def load(self, path, ratio=1.):

        point_cloud_state_dict = torch.load(os.path.join(path, 'point_cloud_state_dict.pt'))
        hyperparameters = torch.load(os.path.join(path, 'hyperparameters.pt'))

        shapes = point_cloud_state_dict["convex_points"]
        max_shape = shapes.shape[0]
        print(f"Loaded {max_shape} convex shapes")
    
        i = 0
        plus = int(max_shape * ratio)

        self._convex_points = point_cloud_state_dict["convex_points"][i:i+plus].to(torch.float32).detach().clone().requires_grad_(True)
        self._delta = point_cloud_state_dict["delta"][i:i+plus].to(torch.float32).detach().clone().requires_grad_(True)
        self._sigma = point_cloud_state_dict["sigma"][i:i+plus].to(torch.float32).detach().clone().requires_grad_(True)
        self.active_sh_degree = point_cloud_state_dict["active_sh_degree"] 
        self._features_dc = point_cloud_state_dict["features_dc"][i:i+plus].to(torch.float32).detach().clone().requires_grad_(True)
        self._features_rest = point_cloud_state_dict["features_rest"][i:i+plus].to(torch.float32).detach().clone().requires_grad_(True)
        self._opacity = point_cloud_state_dict["opacity"][i:i+plus].to(torch.float32).detach().clone().requires_grad_(True)

        self.max_radii2D = hyperparameters["max_radii2D"][i:i+plus].detach().clone().requires_grad_(True)
        self.convex_accumulation_gradient_accum = hyperparameters["convex_accumulation_gradient_accum"][i:i+plus].detach().clone().requires_grad_(True)
        self.denom = hyperparameters["denom"][i:i+plus].detach().clone().requires_grad_(True)
        self.max_scaling = hyperparameters["max_scaling"][i:i+plus].detach().clone().requires_grad_(True)
        self.max_density_factor = hyperparameters["max_density_factor"][i:i+plus].detach().clone().requires_grad_(True)
        self.sigma_accumulation_gradient_accum = hyperparameters["sigma_accumulation_gradient_accum"][i:i+plus].detach().clone().requires_grad_(True)
        
        self._mask = nn.Parameter(torch.ones((self._convex_points.size(0), 1), device="cuda").requires_grad_(True))
        num_points_per_convex = []
        for i in range(self._convex_points.size(0)):
            num_points_per_convex.append(self._convex_points[i].shape[0])
        tensor_num_points_per_convex = torch.tensor(num_points_per_convex, dtype=torch.int, device='cuda:0')
        cumsum_of_points_per_convex = torch.cumsum(torch.nn.functional.pad(tensor_num_points_per_convex, (1,0), value=0), 0, dtype=torch.int)[:-1]
        number_of_points = self._convex_points.shape[0]


        self._num_points_per_convex = tensor_num_points_per_convex
        self._cumsum_of_points_per_convex = cumsum_of_points_per_convex
        self._number_of_points = number_of_points

        l = [
            {'params': [self._features_dc], 'lr': 0.00001, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': 0.00001 / self.feature_ratio, "name": "f_rest"},
            {'params': [self._opacity], 'lr': 0.00001, "name": "opacity"},
            {'params': [self._convex_points], 'lr': 0.00001, "name": "convex_points"},
            {'params': [self._delta], 'lr': 0.00001, "name": "delta"},
            {'params': [self._sigma], 'lr': 0.00001, "name": "sigma"},
            {'params': [self._mask], 'lr':  0.00001, "name": "mask"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def capture(self):
        return (
            self.active_sh_degree,
            self._features_dc,
            self._features_rest,
            self._opacity,
            self.max_radii2D,
            self.convex_accumulation_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._features_dc, 
        self._features_rest,
        self._opacity,
        self.max_radii2D, 
        convex_accumulation_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.convex_accumulation_gradient_accum = convex_accumulation_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    @property
    def get_convex_points_flatten(self):
        return self._convex_points.flatten(0)
  
    @property
    def get_convex_points(self):
        return self._convex_points
    
    @property
    def get_max_scaling(self):
        return self.max_scaling
    
    @property
    def get_delta(self):
        return self.exponential_activation(self._delta)
    
    @property
    def get_sigma(self):
        return self.exponential_activation(self._sigma)
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_num_points_per_convex(self):
        return self._num_points_per_convex
    
    @property
    def get_cumsum_of_points_per_convex(self):
        return self._cumsum_of_points_per_convex
    
    @property
    def get_number_of_points(self):
        return self._number_of_points


    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, opacity : float, init_size : float, nb_points: int, set_delta : float, set_sigma : float, max_grad : float, light: bool):
        
        self.nb_points = nb_points
        self.spatial_lr_scale = spatial_lr_scale

        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())

        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, :, 1:] = torch.randn_like(features[:, :, 1:]) * 1e-4 

        # features[:, 3:, 1:] = 0.0 - old

        x, y, z = fused_point_cloud[:, 0], fused_point_cloud[:, 1], fused_point_cloud[:, 2] # Extract the x, y, z coordinates

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)

        x_min, x_max = torch.min(x), torch.max(x)
        y_min, y_max = torch.min(y), torch.max(y)
        z_min, z_max = torch.min(z), torch.max(z)
        width = x_max - x_min
        height = y_max - y_min
        depth = z_max - z_min
        scene_size = max(width, height, depth)

        self.max_grad = max_grad
        if scene_size > 300 and not light:
            print("Scene is large, we increase the threshold")
            self.max_grad *= 8.5

        
        radii = init_size * torch.sqrt(dist2).unsqueeze(1)

        points_per_convex = fibonacci_sphere(x, y, z, radii, nb_points)

        num_points_per_convex = []
        for i in range(points_per_convex.size(0)):
            num_points_per_convex.append(points_per_convex[i].shape[0])
        tensor_num_points_per_convex = torch.tensor(num_points_per_convex, dtype=torch.int, device='cuda:0')
        cumsum_of_points_per_convex = torch.cumsum(torch.nn.functional.pad(tensor_num_points_per_convex, (1,0), value=0), 0, dtype=torch.int)[:-1]
        number_of_points = points_per_convex.shape[0]
        
        opacities = inverse_sigmoid(opacity * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        deltas = self.inverse_exponential_activation(torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda") * set_delta * (1 / torch.reshape(torch.sqrt(dist2), (-1, 1))))
        sigmas = self.inverse_exponential_activation(torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda") * set_sigma)

        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._convex_points = nn.Parameter(points_per_convex.to('cuda').requires_grad_(True))
        self._delta = nn.Parameter(deltas.requires_grad_(True))
        self._sigma = nn.Parameter(sigmas.requires_grad_(True))
        self._num_points_per_convex = tensor_num_points_per_convex
        self._cumsum_of_points_per_convex = cumsum_of_points_per_convex
        self._number_of_points = number_of_points
        self.max_scaling = torch.zeros((fused_point_cloud.shape[0]), dtype=torch.float, device="cuda")
        self.max_radii2D = torch.zeros((fused_point_cloud.shape[0]), dtype=torch.float, device="cuda")
        self.max_density_factor = torch.zeros((fused_point_cloud.shape[0]), dtype=torch.float, device="cuda")
        self._mask = nn.Parameter(torch.ones((fused_point_cloud.shape[0], 1), device="cuda").requires_grad_(True))

    def training_setup(self, training_args, lr_mask, lr_features, lr_opacity, lr_delta, lr_sigma, lr_convex_points_init, lr_convex_points_final, shifting_cloning, scaling_cloning, sigma_scaling_cloning, delta_scaling_cloning, opacity_cloning):

        self.shifting_cloning = shifting_cloning
        self.scaling_cloning = scaling_cloning
        self.sigma_scaling_cloning = sigma_scaling_cloning
        self.delta_scaling_cloning = delta_scaling_cloning
        self.opacity_cloning = opacity_cloning

        self.convex_accumulation_gradient_accum = torch.zeros((self.get_convex_points.shape[0], 1), device="cuda")
        self.sigma_accumulation_gradient_accum = torch.zeros((self.get_convex_points.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_convex_points.shape[0], 1), device="cuda")


        l = [
            {'params': [self._features_dc], 'lr': lr_features, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': lr_features / self.feature_ratio, "name": "f_rest"},
            {'params': [self._opacity], 'lr': lr_opacity, "name": "opacity"},
            {'params': [self._convex_points], 'lr': lr_convex_points_init, "name": "convex_points"},
            {'params': [self._delta], 'lr': lr_delta, "name": "delta"},
            {'params': [self._sigma], 'lr': lr_sigma, "name": "sigma"},
            {'params': [self._mask], 'lr':  lr_mask, "name": "mask"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.convex_scheduler_args = get_expon_lr_func(lr_init=lr_convex_points_init,
                                                        lr_final=lr_convex_points_final,
                                                        lr_delay_mult=training_args.position_lr_delay_mult,
                                                        max_steps=training_args.position_lr_max_steps)
                                                        
        self.convex_feature_dc_scheduler_args = get_expon_lr_func(lr_init=lr_features,
                                                        lr_final=lr_features / 1000,
                                                        lr_delay_mult=training_args.position_lr_delay_mult,
                                                        max_steps=5000)
                                                        
    
        self.convex_feature_rest_scheduler_args = get_expon_lr_func(lr_init=lr_features / self.feature_ratio,
                                                        lr_final=lr_features / 1000,
                                                        lr_delay_mult=training_args.position_lr_delay_mult,
                                                        max_steps=5000)



    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "convex_points":
                lr = self.convex_scheduler_args(iteration)
                param_group['lr'] = lr
            if iteration < 800: # instead of 1000
                if param_group["name"] == "f_dc":
                    #lr = self.convex_feature_dc_scheduler_args(min(1, iteration - 1000))
                    lr = 0.0075
                    param_group['lr'] = lr
                if param_group["name"] == "f_rest":
                    #lr = self.convex_feature_rest_scheduler_args(min(1, iteration - 1000))
                    lr = 0.0075 / 20
                    param_group['lr'] = lr
            else:
                if param_group["name"] == "f_dc":
                    lr = self.convex_feature_dc_scheduler_args(min(1, iteration - 800))
                    param_group['lr'] = lr
                if param_group["name"] == "f_rest":
                    lr = self.convex_feature_rest_scheduler_args(min(1, iteration - 800))
                    param_group['lr'] = lr
                
        return lr
            
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
    

        return optimizable_tensors
    
    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._convex_points = optimizable_tensors["convex_points"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._delta = optimizable_tensors["delta"]
        self._sigma = optimizable_tensors["sigma"]
        self._mask = optimizable_tensors["mask"]

        self.convex_accumulation_gradient_accum = self.convex_accumulation_gradient_accum[valid_points_mask]
        self.sigma_accumulation_gradient_accum = self.sigma_accumulation_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_scaling = self.max_scaling[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.max_density_factor = self.max_density_factor[valid_points_mask]

        num_points_per_convex = []
        for i in range(self._convex_points.size(0)):
            num_points_per_convex.append(self._convex_points[i].shape[0])
        tensor_num_points_per_convex = torch.tensor(num_points_per_convex, dtype=torch.int, device='cuda:0')
        cumsum_of_points_per_convex = torch.cumsum(torch.nn.functional.pad(tensor_num_points_per_convex, (1,0), value=0), 0, dtype=torch.int)[:-1]
        number_of_points = self._convex_points.shape[0]

        self._num_points_per_convex = tensor_num_points_per_convex
        self._cumsum_of_points_per_convex = cumsum_of_points_per_convex
        self._number_of_points = number_of_points


    def densification_postfix(self, new_convex_points, new_features_dc, new_features_rest, new_opacities, new_delta, new_sigma, new_mask):
        d = {"convex_points": new_convex_points,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "delta" : new_delta,
        "sigma" : new_sigma,
        "mask": new_mask}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._convex_points = optimizable_tensors["convex_points"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._delta = optimizable_tensors["delta"]
        self._sigma = optimizable_tensors["sigma"]
        self._mask = optimizable_tensors["mask"]

        self.convex_accumulation_gradient_accum = torch.zeros((self.get_convex_points.shape[0], 1), device="cuda")
        self.sigma_accumulation_gradient_accum = torch.zeros((self.get_convex_points.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_convex_points.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_convex_points.shape[0]), device="cuda")
        self.max_density_factor = torch.zeros((self.get_convex_points.shape[0]), device="cuda")

        self.max_scaling = torch.cat((self.max_scaling, torch.zeros(new_opacities.shape[0], device="cuda")),dim=0)

        num_points_per_convex = []
        for i in range(self._convex_points.size(0)):
            num_points_per_convex.append(self._convex_points[i].shape[0])
        tensor_num_points_per_convex = torch.tensor(num_points_per_convex, dtype=torch.int, device='cuda:0')
        cumsum_of_points_per_convex = torch.cumsum(torch.nn.functional.pad(tensor_num_points_per_convex, (1,0), value=0), 0, dtype=torch.int)[:-1]
        number_of_points = self._convex_points.shape[0]

        self._num_points_per_convex = tensor_num_points_per_convex
        self._cumsum_of_points_per_convex = cumsum_of_points_per_convex
        self._number_of_points = number_of_points


    def densify_and_split_sigma_big(self, selected_pts_mask):
        
        num_selected = selected_pts_mask.sum().item()

        if num_selected == 0:
            return  # No convex shapes to split

        selected_indices = torch.nonzero(selected_pts_mask).squeeze(1)

        selected_convex_points = self._convex_points[selected_indices]
        centroids = selected_convex_points.mean(dim=1, keepdim=True)  # Shape: [num_selected, 1, 3]
        new_convex_points_list = []


        for i in range(self.nb_points):
            shift_point = selected_convex_points[:, i % selected_convex_points.shape[1], :]  

            shift_vector = (shift_point - centroids.squeeze(1)) * self.shifting_cloning ###### How much we shift
            new_centroid = centroids.squeeze(1) + shift_vector 
            relative_positions = selected_convex_points - centroids  

            scaling_factor = self.scaling_cloning # How much we scale the size
            scaled_relative_positions = scaling_factor * relative_positions

            new_points = new_centroid.unsqueeze(1) + scaled_relative_positions  

            new_convex_points_list.append(new_points)

        new_convex_points = torch.cat(new_convex_points_list, dim=0)


        # Duplicate other attributes
        new_features_dc = self._features_dc[selected_indices].repeat(self.nb_points, 1, 1)
        new_features_rest = self._features_rest[selected_indices].repeat(self.nb_points, 1, 1)
        new_opacities = self._opacity[selected_indices].repeat(self.nb_points, 1)
        new_opacities = inverse_sigmoid(self.opacity_cloning * self.opacity_activation(new_opacities))
        new_delta = self._delta[selected_indices].repeat(self.nb_points, 1) * self.delta_scaling_cloning
        new_sigma = self._sigma[selected_indices].repeat(self.nb_points, 1) * self.sigma_scaling_cloning
        new_mask = self._mask[selected_pts_mask].repeat(self.nb_points,1) 
 
        # Add new convex shapes to the model
        self.densification_postfix(
            new_convex_points,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_delta,
            new_sigma, 
            new_mask
        )

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(self.nb_points * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)


    def densify_and_prune(self, min_opacity, mask_threshold, extent, max_screen_size):

        grads_sigma = self.sigma_accumulation_gradient_accum / self.denom
        grads_sigma[grads_sigma.isnan()] = 0.0
        # SIGMA DENSIFICATION
        grads_sigma = grads_sigma.squeeze()
        selected_pts_mask = torch.where(grads_sigma >= self.max_grad, True, False)
        self.densify_and_split_sigma_big(selected_pts_mask)

        prune_mask = torch.logical_or((torch.sigmoid(self._mask) <= mask_threshold).squeeze(),(self.get_opacity < min_opacity).squeeze())
        if max_screen_size:
            big_points_ws = self.get_max_scaling > max_screen_size * extent
            prune_mask = torch.logical_or(prune_mask, big_points_ws)

        self.prune_points(prune_mask)
        torch.cuda.empty_cache()


    def only_prune(self, min_opacity, mask_threshold):
        prune_mask = torch.logical_or((torch.sigmoid(self._mask) <= mask_threshold).squeeze(),(self.get_opacity < min_opacity).squeeze())
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def reset_opacity(self, sigma_reset):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*sigma_reset))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def add_densification_stats(self, viewspace_point_tensor, viewspace_sigma, update_filter, scaling, density_factor):
        self.convex_accumulation_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        self.max_scaling[update_filter] = scaling[update_filter]
        self.max_density_factor[update_filter] += density_factor[update_filter]
        self.sigma_accumulation_gradient_accum[update_filter] += viewspace_sigma.grad[update_filter]
