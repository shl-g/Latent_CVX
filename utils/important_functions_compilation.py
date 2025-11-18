import numpy as np
import open3d as o3d
import collections


# ---- PCD and Cameras Block ----- Beginning

def create_pcd_around_origin(N):
    directions = np.random.randn(N, 3)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = np.random.uniform(0.2, 1.0, size=(N, 1))
    points = directions * radii
    colors = np.random.rand(N, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def look_at(origin, target, up=np.array([0, 0, 1], dtype=np.float32)): # <--- This is the look_at_verified function!
    forward = target - origin
    forward = forward / np.linalg.norm(forward)

    # Make up perpendicular to forward
    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)
    new_up = np.cross(forward, right)
    new_up = new_up / np.linalg.norm(new_up)

    R = np.stack([right, new_up, forward], axis=1)
    return R

def fibonacci_sphere(samples=1, radius=1.0): # <--- This is the fib_sphere without points in the poles!
    points = []
    phi = np.pi * (3. - np.sqrt(5.))
    # Offset to avoid poles by skipping very first and last
    for i in range(1, samples + 1):  # Now i in [1, samples]
        y = 1 - (i / float(samples + 1)) * 2  # Avoid y = ±1
        r = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * r
        z = np.sin(theta) * r
        points.append([x * radius, y * radius, z * radius])
    return np.array(points)

def get_view_direction(thetas, phis, overhead=30, front=60): # <--- Modified a bit from latent-NeRF
    #                   phis [B,];          thetas: [B,]
    # front = 0         [0, front)
    # side (left) = 1   [front, 180)
    # back = 2          [180, 180+front)
    # side (right) = 3  [180+front, 360)
    # top = 4                               [0, overhead]
    # bottom = 5                            [180-overhead, 180]

    overhead = np.deg2rad(overhead)
    front = np.deg2rad(front)

    res = torch.zeros(thetas.shape[0], dtype=torch.long)
    # first determine by phis

    # res[(phis < front)] = 0
    res[(phis >= (2 * np.pi - front / 2)) & (phis < front / 2)] = 0

    # res[(phis >= front) & (phis < np.pi)] = 1
    res[(phis >= front / 2) & (phis < (np.pi - front / 2))] = 1

    # res[(phis >= np.pi) & (phis < (np.pi + front))] = 2
    res[(phis >= (np.pi - front / 2)) & (phis < (np.pi + front / 2))] = 2

    # res[(phis >= (np.pi + front))] = 3
    res[(phis >= (np.pi + front / 2)) & (phis < (2 * np.pi - front / 2))] = 3
    # override by thetas
    res[thetas <= overhead] = 4
    res[thetas >= (np.pi - overhead)] = 5
    return res

def generate_cameras_around_point(center=[0,0,0], num_cameras=3, radius=2.0, image_size=(800, 800), fov_deg=40):
    camera_list = []
    width, height = image_size
    fov_rad = np.deg2rad(fov_deg)
    fx = fy = 0.5 * width / np.tan(fov_rad / 2)
    cx, cy = width / 2, height / 2
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])

    # Generate camera positions on a sphere around the new center
    sphere_points = fibonacci_sphere(num_cameras, radius)
    cam_positions = sphere_points + np.array(center)

    for i, cam_pos in enumerate(cam_positions):
        # Create look-at rotation matrix pointing to the center
        R_cam_to_world = look_at(cam_pos, np.array(center))

        # Compute world-to-camera extrinsics
        R_world_to_cam = np.linalg.inv(R_cam_to_world)
        t_world_to_cam = -R_world_to_cam @ cam_pos

        extrinsics = np.eye(4)
        extrinsics[:3, :3] = R_world_to_cam
        extrinsics[:3, 3] = t_world_to_cam

        camera_list.append({
            'intrinsics': K,
            'extrinsics': extrinsics,
            'position': cam_pos,
            'camera_id': i
        })

    return camera_list

def generate_cameras_around_point_with_dir(center=[0,0,0], num_cameras=3, radius=2.0, image_size=(800, 800), fov_deg=40):
    camera_list = []
    width, height = image_size
    fov_rad = np.deg2rad(fov_deg)
    fx = fy = 0.5 * width / np.tan(fov_rad / 2)
    cx, cy = width / 2, height / 2
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])

    # Generate camera positions on a sphere around the new center
    sphere_points = fibonacci_sphere(num_cameras, radius)
    cam_positions = sphere_points + np.array(center)

    for i, cam_pos in enumerate(cam_positions):
        # Create look-at rotation matrix pointing to the center
        R_cam_to_world = look_at(cam_pos, np.array(center))

        # Compute world-to-camera extrinsics
        R_world_to_cam = np.linalg.inv(R_cam_to_world)
        t_world_to_cam = -R_world_to_cam @ cam_pos

        extrinsics = np.eye(4)
        extrinsics[:3, :3] = R_world_to_cam
        extrinsics[:3, 3] = t_world_to_cam

        theta = np.arctan(cam_pos[1] / cam_pos[0])
        phi = np.arctan(np.sqrt((cam_pos[0] ** 2) + (cam_pos[1] ** 2)) / cam_pos[2])
        direction = get_view_direction(theta , phi)

        camera_list.append({
            'intrinsics': K,
            'extrinsics': extrinsics,
            'position': cam_pos,
            'camera_id': i,
            'dir': direction
        })

    return camera_list

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

# ---- Camera Data Structures ----
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"]) # Intrinsics Structure
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]) # Extrinsics Structure
# --------------------------------

def get_intrinsics(camera_id, intr_data, image_size):
    model = "PINHOLE"
    fx = intr_data[0, 0]
    fy = intr_data[1, 1]
    cx = intr_data[0, 2]
    cy = intr_data[1, 2]
    width, height = image_size
    params = np.array([fx, fy, cx, cy])

    camera_intr = Camera(id=camera_id, model=model,
                         width=width, height=height,
                         params=params)
    return camera_intr

def extract_extrinsics(extrinsics):
    """
    Extract extrinsics in COLMAP format: [qvec, tvec]
    qvec: [QW, QX, QY, QZ]
    tvec: [TX, TY, TZ]
    """
    R_mat = extrinsics[:3, :3]
    tvec = extrinsics[:3, 3]
    qvec = rotmat2qvec(R_mat)

    return [qvec, tvec]

def get_extrinsics(image_id, extr_data, image_size):
    qvec, tvec = extract_extrinsics(extr_data)
    camera_id = image_id
    image_name = None
    xys = None
    point3D_ids = None

    camera_extr = BaseImage(
        id=image_id, qvec=qvec, tvec=tvec,
        camera_id=camera_id, name=image_name,
        xys=xys, point3D_ids=point3D_ids)

    return camera_extr

# ---- Cameras Reading and Initialization Functions (Not complete, very W.I.P., based on the CVXS code, didn't import everything) ----

# def readO3DCameras(cam_extrinsics, cam_intrinsics, images_folder):
#     image_files = sorted([f for f in os.listdir(images_folder)
#                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
#     cam_infos = []
#     for idx, key in enumerate(cam_extrinsics):
#         sys.stdout.write('\r')
#         # the exact output you're looking for:
#         sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
#         sys.stdout.flush()
#
#         extr = cam_extrinsics[key]
#         intr = cam_intrinsics[extr.camera_id]
#         height = intr.height
#         width = intr.width
#
#         uid = intr.id
#         R = np.transpose(qvec2rotmat(extr.qvec))
#         T = np.array(extr.tvec)
#
#         if intr.model == "SIMPLE_PINHOLE":
#             focal_length_x = intr.params[0]
#             FovY = focal2fov(focal_length_x, height)
#             FovX = focal2fov(focal_length_x, width)
#         elif intr.model == "PINHOLE":
#             focal_length_x = intr.params[0]
#             focal_length_y = intr.params[1]
#             FovY = focal2fov(focal_length_y, height)
#             FovX = focal2fov(focal_length_x, width)
#         else:
#             assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
#
#         image_path = os.path.join(images_folder, image_files[idx])
#         image_name = os.path.basename(image_path).split(".")[0]
#         image = Image.open(image_path)
#
#         cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
#                               image_path=image_path, image_name=image_name, width=width, height=height)
#         cam_infos.append(cam_info)
#     sys.stdout.write('\n')
#     return cam_infos
#
# def readOpen3DSceneInfo(path, images, eval, llffhold=8):
#     num_cameras = 69
#     image_size = (800, 800)
#     generated_cameras = generate_cameras_around_origin(num_cameras, radius=3.0, image_size=image_size)
#     cam_intrinsics = {}
#     cam_extrinsics = {}
#     for camera in generated_cameras:
#         intr_data = camera["intrinsics"]
#         extr_data = camera["extrinsics"]
#         camera_id = camera["camera_id"]
#         cam_pos = camera["position"]
#
#         cam_intr = get_intrinsics(camera_id, intr_data, image_size)
#         cam_extr = get_extrinsics(camera_id, extr_data, image_size)
#
#         cam_intrinsics[camera_id] = cam_intr
#         cam_extrinsics[camera_id] = cam_extr
#
#
#     reading_dir = "images" if images == None else images
#     cam_infos_unsorted = readO3DCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
#                                            images_folder=os.path.join(path, reading_dir))
#     cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)
#
#     """We might not need this eval part"""
#     if eval:
#         train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
#         test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
#     else:
#         train_cam_infos = cam_infos
#         test_cam_infos = []
#
#     nerf_normalization = getNerfppNorm(train_cam_infos)
#
#     ply_path = os.path.join(path, "sparse/0/points3D.ply")
#     bin_path = os.path.join(path, "sparse/0/points3D.bin")
#     txt_path = os.path.join(path, "sparse/0/points3D.txt")
#     if not os.path.exists(ply_path):
#         print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
#         try:
#             xyz, rgb, _ = read_points3D_binary(bin_path)
#         except:
#             try:
#                 xyz, rgb, _ = read_points3D_text(txt_path)
#             except:
#                 num_points = 3000
#                 pcd = create_point_cloud(num_points)
#                 xyz, rgb = np.asarray(pcd.points), np.asarray(pcd.colors)
#
#         storePly(ply_path, xyz, rgb)
#     try:
#         pcd = fetchPly(ply_path)
#     except:
#         pcd = None
#
#     scene_info = SceneInfo(point_cloud=pcd,
#                            train_cameras=train_cam_infos,
#                            test_cameras=test_cam_infos,
#                            nerf_normalization=nerf_normalization,
#                            ply_path=ply_path)
#     return scene_info
#
# def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
#     cam_infos = []
#     for idx, key in enumerate(cam_extrinsics):
#         sys.stdout.write('\r')
#         # the exact output you're looking for:
#         sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
#         sys.stdout.flush()
#
#         extr = cam_extrinsics[key]
#         intr = cam_intrinsics[extr.camera_id]
#         height = intr.height
#         width = intr.width
#
#         uid = intr.id
#         R = np.transpose(qvec2rotmat(extr.qvec))
#         T = np.array(extr.tvec)
#
#         if intr.model == "SIMPLE_PINHOLE":
#             focal_length_x = intr.params[0]
#             FovY = focal2fov(focal_length_x, height)
#             FovX = focal2fov(focal_length_x, width)
#         elif intr.model == "PINHOLE":
#             focal_length_x = intr.params[0]
#             focal_length_y = intr.params[1]
#             FovY = focal2fov(focal_length_y, height)
#             FovX = focal2fov(focal_length_x, width)
#         else:
#             assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
#
#         image_path = os.path.join(images_folder, os.path.basename(extr.name))
#         image_name = os.path.basename(image_path).split(".")[0]
#         image = Image.open(image_path)
#
#         cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
#                               image_path=image_path, image_name=image_name, width=width, height=height)
#         cam_infos.append(cam_info)
#     sys.stdout.write('\n')
#     return cam_infos
#
# def readColmapSceneInfo_debug(path, images, eval, llffhold=8):
#
#     cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
#     cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
#     cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
#     cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
#
#     # debug section
#     num_cameras = 100000
#     image_size = (800, 800)
#     generated_cameras = generate_cameras_around_point([-0.5, 1, 2.5],num_cameras, radius=4.37, image_size=image_size)
#     inters_from_gen = {}
#     exters_from_gen = {}
#     for camera in generated_cameras:
#         intr_data = camera["intrinsics"]
#         extr_data = camera["extrinsics"]
#         camera_id = camera["camera_id"]
#         cam_pos = camera["position"]
#
#         cam_intr = get_intrinsics(camera_id, intr_data, image_size)
#         cam_extr = get_extrinsics_debug(camera_id, extr_data, image_size)
#
#         inters_from_gen[camera_id] = cam_intr
#         exters_from_gen[camera_id] = cam_extr
#
#     for cam in generated_cameras[:5]:
#         R = cam['extrinsics'][:3, :3]
#         t = cam['extrinsics'][:3, 3]
#         center_ray = -R.T[:, 2]  # direction of view in world coords
#         norm = np.linalg.norm(np.array([-0.5, 1, 2.5]) - cam['position'])
#         to_center = (np.array([-0.5, 1, 2.5]) - cam['position']) / norm
#         print("Dot(center_ray, to_center) = ", np.dot(center_ray, to_center))
#     print("extr matching start 28")
#     match_and_update_by_position(cam_extrinsics, exters_from_gen, output_path="matched_cameras.txt")
#     #print("intr matching start")
#     #match_and_update_intrinsics(cam_intrinsics, inters_from_gen)
#     #end of debug
#
#     reading_dir = "images" if images == None else images
#     cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
#                                            images_folder=os.path.join(path, reading_dir))
#     cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)
#
#     for i in range(3):
#         print(cam_infos[i])
#
#     if eval:
#         train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
#         test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
#     else:
#         train_cam_infos = cam_infos
#         test_cam_infos = []
#
#     nerf_normalization = getNerfppNorm(train_cam_infos)
#
#     ply_path = os.path.join(path, "sparse/0/points3D.ply")
#     bin_path = os.path.join(path, "sparse/0/points3D.bin")
#     txt_path = os.path.join(path, "sparse/0/points3D.txt")
#     num_points = 3000
#     pcd = create_point_cloud_shifted(num_points)
#     xyz, rgb = np.asarray(pcd.points), np.asarray(pcd.colors)
#     storePly(ply_path, xyz, rgb)
#     pcd = fetchPly(ply_path)
#     # if not os.path.exists(ply_path):
#     #     print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
#     #     try:
#     #         xyz, rgb, _ = read_points3D_binary(bin_path)
#     #     except:
#     #         try:
#     #             xyz, rgb, _ = read_points3D_text(txt_path)
#     #         except:
#     #
#     #
#     #     storePly(ply_path, xyz, rgb)
#     # try:
#     #
#     # except:
#     #     pcd = None
#
#     scene_info = SceneInfo(point_cloud=pcd,
#                            train_cameras=train_cam_infos,
#                            test_cameras=test_cam_infos,
#                            nerf_normalization=nerf_normalization,
#                            ply_path=ply_path)
#     return scene_info

# ------------------------------------------------------------------------------------------------------------------------------------

# ---- PCD and Cameras Block ----- End



# ---- Stable Diffusion Block ---- Beginning

from huggingface_hub import hf_hub_download
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel, logging, CLIPProcessor
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F

import time


class StableDiffusion(nn.Module):
    def __init__(self, device, model_name='CompVis/stable-diffusion-v1-4', concept_name=None, latent_mode=True):
        super().__init__()

        try:
            with open('./TOKEN', 'r') as f:
                self.token = f.read().replace('\n', '')  # remove the last \n!
                print(f'loaded hugging face access token from ./TOKEN!')
        except FileNotFoundError as e:
            self.token = True
            print(f'try to load hugging face access token from the default place, make sure you have run `huggingface-cli login`.')

        self.device = device
        self.latent_mode = latent_mode
        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.98)

        print(f'loading stable diffusion with {model_name}...')

        # 1. Load the autoencoder model which will be used to decode the latents into image space.
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae", use_auth_token=self.token).to(self.device)

        # 2. Load the tokenizer and text encoder to tokenize and encode the text.
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.image_encoder = None
        self.image_processor = None

        # 3. The UNet model for generating the latents.
        self.unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet", use_auth_token=self.token).to(
            self.device)

        # 4. Create a scheduler for inference
        self.scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                       num_train_timesteps=self.num_train_timesteps)
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        if concept_name is not None:
            self.load_concept(concept_name)
        print(f'\t successfully loaded stable diffusion!')

    def load_concept(self, concept_name):
        repo_id_embeds = f"sd-concepts-library/{concept_name}"
        learned_embeds_path = hf_hub_download(repo_id=repo_id_embeds, filename="learned_embeds.bin")
        token_path = hf_hub_download(repo_id=repo_id_embeds, filename="token_identifier.txt")
        with open(token_path, 'r') as file:
            placeholder_token_string = file.read()

        loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")

        # separate token and the embeds
        trained_token = list(loaded_learned_embeds.keys())[0]
        embeds = loaded_learned_embeds[trained_token]

        # cast to dtype of text_encoder
        dtype = self.text_encoder.get_input_embeddings().weight.dtype
        embeds.to(dtype)

        # add the token in tokenizer
        token = trained_token
        num_added_tokens = self.tokenizer.add_tokens(token)
        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the token {token}. Please pass a different `token` that is not already in the tokenizer.")

        # resize the token embeddings
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        # get the id for the token and assign the embeds
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        self.text_encoder.get_input_embeddings().weight.data[token_id] = embeds

    def get_text_embeds(self, prompt):
        # Tokenize text and get embeddings
        pos_prompt = f'a highly detailed {prompt}'
        text_input_pos = self.tokenizer(pos_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')

        with torch.no_grad():
            text_embeddings_pos = self.text_encoder(text_input_pos.input_ids.to(self.device))[0]
        
        neg_prompt = f'unrealistic, blurry, low quality, out of focus, ugly, low contrast, dull, dark, low-resolution, gloomy {prompt}'
        text_input_neg = self.tokenizer(neg_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
        
        with torch.no_grad():
            text_embeddings_neg = self.text_encoder(text_input_neg.input_ids.to(self.device))[0]
            
        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer([''] * len(prompt), padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings_pos, text_embeddings_neg])
        return text_embeddings
    
    

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents
        
    def train_step(self, text_embeddings, inputs, guidance_scale=100):

        # interp to 512x512 to be fed into vae.

        """# _t = time.time()
        #print("latent mode = ", self.latent_mode)
        if not self.latent_mode:
            # latents = F.interpolate(latents, (64, 64), mode='bilinear', align_corners=False)
            pred_rgb_512 = F.interpolate(inputs, (512, 512), mode='bilinear', align_corners=False)
            latents = self.encode_imgs(pred_rgb_512)
        else:
            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(0)
            #latents = self.encode_imgs(inputs)
            latents = inputs"""
            
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)    
        latents = self.encode_imgs(inputs)
        
        # torch.cuda.synchronize(); print(f'[TIME] guiding: interp {time        .time() - _t:.4f}s')

        # After encoding
        #print("Latents shape:", latents.shape, "Requires grad:", latents.requires_grad)
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)
        
        # encode image into latents with vae, requires grad!
        # _t = time.time()

        # torch.cuda.synchronize(); print(f'[TIME] guiding: vae enc {time.time() - _t:.4f}s')

        # predict the noise residual with unet, NO grad!
        # _t = time.time()
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            if t< 200:
                latent_model_input = torch.cat([latents_noisy] * 2)
                text_embeddings = text_embeddings[0:2]
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                DeltaD = noise_pred_uncond
            else:
                latent_model_input = torch.cat([latents_noisy] * 3)
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                noise_pred_uncond, noise_pred_text, noise_pred_neg = noise_pred.chunk(3)
                DeltaD = noise_pred_uncond - noise_pred_neg

        sDeltaC = guidance_scale * (noise_pred_text - noise_pred_uncond)
        noise_pred = noise_pred_uncond + sDeltaC # Changed from: noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: unet {time.time() - _t:.4f}s')

        # perform guidance (high scale from paper!)


        # w(t), alpha_t * sigma_t^2
        # w = (1 - self.alphas[t])
        w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])

        # After grad
        #print("Grad stats:", grad.min().item(), grad.max().item(), grad.mean().item())
        # clip grad for stable training?
        # grad = grad.clamp(-1, 1)

        # manually backward, since we omitted an item in grad and cannot simply autodiff.
        # _t = time.time()
        #print("Latents requires_grad:", latents.requires_grad)
        use_NFSD = True
        if use_NFSD == True:
            NFSD = w * (DeltaD + sDeltaC) # Moved inside the if_else clause
            latents.backward(gradient=NFSD, retain_graph=True)
        else:
            SDS = w * (noise_pred - noise) # Moved inside the if_else clause
            latents.backward(gradient=SDS, retain_graph=True)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: backward {time.time() - _t:.4f}s')

        return torch.tensor(0.0, requires_grad=True, device=self.device)
        # dummy loss value used to be 0

def calc_text_embeddings(ref_text, diff_model):
    text_z_list = []
    for d in ['front', 'side', 'back', 'side', 'overhead', 'bottom']:
        text = f"{ref_text}, {d} view"
        text_z_list.append(diff_model.get_text_embeds([text]))
    return text_z_list

def RGB_to_latent(image):
    """
    Projects an RGB image [3, H, W] to a 4-channel latent [1, 4, H, W]
    using a fixed linear transformation.
    """
    assert image.dim() == 3 and image.shape[0] == 3, "Expected image of shape [3, H, W]"

    weights = torch.tensor([
        [0.298,  0.207,  0.208],   # L1
        [0.187,  0.286,  0.173],   # L2
        [-0.158, 0.189,  0.264],   # L3
        [-0.184, -0.271, -0.473],  # L4
    ], device=image.device, dtype=image.dtype)  # [4, 3]

    H, W = image.shape[1], image.shape[2]  # get height and width
    image_flat = image.view(3, -1)         # [3, H*W]
    latent_flat = weights @ image_flat     # [4, H*W]
    latent = latent_flat.view(1, 4, H, W)  # [1, 4, H, W]

    return latent
