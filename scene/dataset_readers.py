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
import sys
import collections
import random
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
import open3d as o3d
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.convex_model import BasicPointCloud

# ---- Open3D Cameras Functions ----

Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

def create_pcd_around_origin(N):
    directions = np.random.randn(N, 3)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = np.random.uniform(0.0, 2.3, size=(N, 1)) #instead of 1.5
    points = directions * radii
    
    greyscale = False
    if greyscale == True:
        grey_values = np.random.uniform(0, 1, size=(N, 1))
        colors = np.repeat(grey_values, 3, axis=1)  # (N, 1) → (N, 3)
    else:
        colors = np.random.normal(loc=0.5, scale=0.15, size=(N, 3))
        colors /= np.linalg.norm(colors, axis=1, keepdims=True) + 1e-6
        colors = 0.5 + 0.5 * colors  # Rescale to [0,1]
        colors = np.clip(colors, 0, 1)

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

def ring_around_origin(num_cameras, radius=4.5):
    angles = np.linspace(0, 2 * np.pi, num_cameras, endpoint=False)
    x = np.cos(angles) * radius
    y = np.sin(angles) * radius
    z = np.zeros_like(x)
    return np.stack([x, y, z], axis=1)

def get_view_direction(thetas, phis, overhead=30, front=90): # <--- Modified a bit from latent-NeRF
    #                   phis [B,];          thetas: [B,]
    # front = 0         [0, front)
    # side (left) = 1   [front, 180)
    # back = 2          [180, 180+front)
    # side (right) = 3  [180+front, 360)
    # top = 4                               [0, overhead]
    # bottom = 5                            [180-overhead, 180]

    overhead = np.deg2rad(overhead)
    front = np.deg2rad(front)

    res = 0
    # first determine by phis

    # res[(phis < front)] = 0
    if (phis >= (- front / 2)) & (phis < front / 2):
        res = 0

    # res[(phis >= front) & (phis < np.pi)] = 1
    if (phis >= front / 2) & (phis < (np.pi - front / 2)):
        res = 1

    # res[(phis >= np.pi) & (phis < (np.pi + front))] = 2
    if (phis >= (np.pi - front / 2)) or (phis < (-np.pi + front / 2)):
        res = 2

    if (phis >= (-np.pi + front / 2)) & (phis < (- front / 2)):
        res = 3

    # override by thetas
    if thetas <= overhead:
        res = 4

    if thetas >= (np.pi - overhead):
        res = 5
    return res

def generate_cameras_around_point(center=[0,0,0], num_cameras=3, radius=4.5, image_size=(512, 512), fov_deg=100):
    camera_list = []
    width, height = image_size

    # Generate camera positions on a sphere around the new center
    sphere_points = ring_around_origin(num_cameras, radius)
    cam_positions = sphere_points + np.array(center)

    for i, cam_pos in enumerate(cam_positions):
        fov_rad = np.deg2rad(fov_deg)
        fx = fy = 0.5 * width / np.tan(fov_rad / 2)
        cx, cy = width / 2, height / 2
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]])

        # Create look-at rotation matrix pointing to the center
        R_cam_to_world = look_at(cam_pos, np.array(center))

        # Compute world-to-camera extrinsics
        R_world_to_cam = np.linalg.inv(R_cam_to_world)
        t_world_to_cam = -R_world_to_cam @ cam_pos

        extrinsics = np.eye(4)
        extrinsics[:3, :3] = R_world_to_cam
        extrinsics[:3, 3] = t_world_to_cam

        r = np.linalg.norm(cam_pos)
        x = cam_pos[0]
        y = cam_pos[1]
        z = cam_pos[2]
        phi = np.arctan2(y, x)  # azimuth angle
        theta = np.arccos(z / r)
        direction = get_view_direction(theta, phi)
        
        # remove top and bottom cams
        if direction != 4 and direction != 5:
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

# ----------------------------------

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    dir: int


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readO3DCameras(cam_extrinsics, cam_intrinsics, cam_directions, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width
        direction = cam_directions[extr.camera_id]

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Camera Intrinsics Model is not PINHOLE - check camera intrinsics generation or fetching"

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=None,
                              image_path=None, image_name=None, width=width, height=height, dir=direction)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readOpen3DSceneInfo(path, images, eval, llffhold=8, num_points=25_000, num_cameras=128):
    image_size = (512, 512)
    generated_cameras = generate_cameras_around_point([0, 0, 0], num_cameras, 7, image_size, 70) #rad was 3. Now 5 for train, 7 for test.
    cam_intrinsics = {}
    cam_extrinsics = {}
    cam_directions = {}
    for camera in generated_cameras:
        intr_data = camera["intrinsics"]
        extr_data = camera["extrinsics"]
        camera_id = camera["camera_id"]
        direction = camera["dir"]

        cam_intr = get_intrinsics(camera_id, intr_data, image_size)
        cam_extr = get_extrinsics(camera_id, extr_data, image_size)

        cam_intrinsics[camera_id] = cam_intr
        cam_extrinsics[camera_id] = cam_extr
        cam_directions[camera_id] = direction


    reading_dir = "images" if images == None else images

    cam_infos_unsorted = readO3DCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                        cam_directions=cam_directions, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.uid)

    # If Else block may need revision
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3D.ply")
    pcd = create_pcd_around_origin(num_points)
    xyz, rgb = np.asarray(pcd.points), np.asarray(pcd.colors)
    storePly(ply_path, xyz, rgb)
    pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                           images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    for i in range(3):
        print(cam_infos[i])

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info



def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                        image_path=image_path, image_name=image_name, width=image.size[0],
                                        height=image.size[1]))

    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


sceneLoadTypeCallbacks = {
    "Open3D": readOpen3DSceneInfo,
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo
}