import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import collections

Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])


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


def create_point_cloud(N):
    directions = np.random.randn(N, 3)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = np.random.uniform(0.2, 1.0, size=(N, 1))
    points = directions * radii
    colors = np.random.rand(N, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def create_point_cloud_shifted(N):
    directions = np.random.randn(N, 3)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = np.random.uniform(0.2, 1.0, size=(N, 1))
    points = directions * radii
    points += np.array([-0.5, 1, 2.5])
    colors = np.random.rand(N, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def look_at(origin, target, up=np.array([0, 1, 0], dtype=np.float32)):
    forward = target - origin
    forward = forward / np.linalg.norm(forward)

    # Handle case where forward and up are parallel
    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)

    if right_norm < 1e-6:  # forward and up are nearly parallel
        # Choose a different up vector
        if abs(forward[1]) < 0.9:
            up = np.array([0, 1, 0], dtype=np.float32)
        else:
            up = np.array([1, 0, 0], dtype=np.float32)
        right = np.cross(forward, up)
        right_norm = np.linalg.norm(right)

    right = right / right_norm
    new_up = np.cross(right, forward)
    new_up = new_up / np.linalg.norm(new_up)
    # Create rotation matrix (camera coordinate system)
    # In camera coordinates: right=+X, up=+Y, forward=-Z
    R = np.stack([right, new_up, -forward], axis=1)
    return R


def fibonacci_sphere(samples=1, radius=1.0):
    points = []
    phi = np.pi * (3. - np.sqrt(5.))
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2
        r = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * r
        z = np.sin(theta) * r
        points.append([x * radius, y * radius, z * radius])
    return np.array(points)


def generate_cameras_around_origin(num_cameras, radius=2.0, image_size=(800, 800), fov_deg=60):
    camera_list = []
    width, height = image_size
    fov_rad = np.deg2rad(fov_deg)
    fx = fy = 0.5 * width / np.tan(fov_rad / 2)
    cx, cy = width / 2, height / 2
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])

    cam_positions = fibonacci_sphere(num_cameras, radius)

    for i, cam_pos in enumerate(cam_positions):
        # Create look-at rotation matrix
        R_cam_to_world = look_at(cam_pos, np.zeros(3))

        # For COLMAP format, we need world-to-camera transformation
        R_world_to_cam = R_cam_to_world
        t_world_to_cam = -R_world_to_cam @ cam_pos

        # Create extrinsics matrix (world-to-camera)
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

def generate_cameras_around_point(center, num_cameras, radius=2.0, image_size=(800, 800), fov_deg=40):
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
        R_world_to_cam = R_cam_to_world
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


def create_camera_frustum(K, extrinsics, image_size=(800, 800), near=0.1, scale=0.3, color=[1, 0, 0]):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    width, height = image_size
    corners_px = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ])
    corners_cam = []
    for u, v in corners_px:
        x = (u - cx) / fx * near
        y = (v - cy) / fy * near
        corners_cam.append([x, y, near])
    points = np.array([[0, 0, 0]] + corners_cam) * scale
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    points_world = (R @ points.T).T + t
    lines = [[0, 1], [0, 2], [0, 3], [0, 4],
             [1, 2], [2, 3], [3, 4], [4, 1]]
    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(points_world)
    frustum.lines = o3d.utility.Vector2iVector(lines)
    frustum.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return frustum


def extract_intrinsics_format(K, image_size):
    """Extract intrinsics in the format: [width, height, params]"""
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    width, height = image_size
    params = np.array([fx, fy, cx, cy])
    return [width, height, params]


def extract_extrinsics_format(extrinsics):
    """
    Extract extrinsics in COLMAP format: [qvec, tvec]
    qvec: [QW, QX, QY, QZ] (quaternion with w first)
    tvec: [TX, TY, TZ] (translation)
    """
    # Extract rotation matrix and translation from extrinsics
    R_mat = extrinsics[:3, :3]
    tvec = extrinsics[:3, 3]

    # Ensure the rotation matrix is valid (orthogonal with determinant 1)
    U, s, Vt = np.linalg.svd(R_mat)
    R_clean = U @ Vt

    # Ensure proper rotation (det = 1, not -1)
    if np.linalg.det(R_clean) < 0:
        U[:, -1] *= -1
        R_clean = U @ Vt

    try:
        # Convert rotation matrix to quaternion using scipy
        # scipy returns [x, y, z, w], but COLMAP uses [w, x, y, z]
        scipy_quat = R.from_matrix(R_clean).as_quat()
        qvec = np.array([scipy_quat[3], scipy_quat[0], scipy_quat[1], scipy_quat[2]])  # [w, x, y, z]
    except Exception as e:
        print(f"Warning: Failed to convert rotation matrix to quaternion: {e}")
        print(f"R_mat determinant: {np.linalg.det(R_mat)}")
        print(f"R_clean determinant: {np.linalg.det(R_clean)}")
        # Fallback to identity quaternion
        qvec = np.array([1.0, 0.0, 0.0, 0.0])

    return [qvec, tvec]

def extract_extrinsics_format_new_ver(extrinsics):
    """
    Extract extrinsics in COLMAP format: [qvec, tvec]
    qvec: [QW, QX, QY, QZ]
    tvec: [TX, TY, TZ]
    """
    R_mat = extrinsics[:3, :3]
    tvec = extrinsics[:3, 3]

    # Orthonormalize rotation matrix using SVD
    U, _, Vt = np.linalg.svd(R_mat)
    R_clean = U @ Vt

    # Ensure proper rotation (det = +1)
    if np.linalg.det(R_clean) < 0:
        U[:, -1] *= -1
        R_clean = U @ Vt

    # Convert to quaternion using your function
    qvec = rotmat2qvec(R_clean)

    return [qvec, tvec]

# taken from colmap_loader.py
def read_extrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = BaseImage(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=None, point3D_ids=None)
    return images


# ---- Run everything ----
N = 3000
num_cameras = 1000
image_size = (800, 800)

# Create point cloud
pcd = create_point_cloud_shifted(N)
#pcd = create_point_cloud(N)

#ply_lego = o3d.io.read_point_cloud("points3d.ply")

# Generate cameras
center = [-0.5, 1, 2.5]
cameras = generate_cameras_around_point(center, num_cameras, radius=2.25, image_size=image_size)
# cameras = generate_cameras_around_origin(num_cameras, radius=2.0, image_size=image_size)
#
#
# # Create frustums for visualization
# frustums = [
#     create_camera_frustum(cam['intrinsics'], np.linalg.inv(cam['extrinsics']),
#                           image_size=image_size, scale=0.5, color=[1, 0, 0])
#     for cam in cameras
# ]
#
# # Show point cloud and camera frustums
# o3d.visualization.draw_geometries([pcd] + frustums, window_name="PointCloud + Camera Frustums")
#o3d.visualization.draw_geometries([pcd, ply_lego], window_name="PointCloud + lego pcd")


# Extract formatted parameters
# print("=== INTRINSICS (same for all cameras) ===")
# intr = extract_intrinsics_format(cameras[0]['intrinsics'], image_size)
# print(f"width = {intr[0]}")
# print(f"height = {intr[1]}")
# print(f"params = {intr[2]}")  # [fx, fy, cx, cy]

# print("=== EXTRINSICS (first 3 cameras) ===")
# for i in range(min(3, len(cameras))):
#     qvec, tvec = extract_extrinsics_format(cameras[i]['extrinsics'])
#     print(f"Camera {i}:")
#     print(f"  qvec = {qvec}")  # [QW, QX, QY, QZ]
#     print(f"  tvec = {tvec}")  # [TX, TY, TZ]
#     print()

# # Optional: Save all formatted data
# print("=== ALL CAMERAS DATA ===")
# all_cameras_data = []
# for i, cam in enumerate(cameras):
#     qvec, tvec = extract_extrinsics_format(cam['extrinsics'])
#     all_cameras_data.append({
#         'camera_id': i,
#         'qvec': qvec,
#         'tvec': tvec,
#         'intrinsics': intr
#     })
#
# print(f"Generated {len(all_cameras_data)} cameras with proper COLMAP format")
# print("Quaternion format: [QW, QX, QY, QZ] (scalar first)")
# print("Translation format: [TX, TY, TZ]")

# Optional: Save data
# np.savez("camera_params.npz",
#          intrinsics=intr,
#          cameras_data=all_cameras_data)
# o3d.io.write_point_cloud("pointcloud.ply", pcd)


def save_pcd_as_full_colmap_format_txt(pcd: o3d.geometry.PointCloud, filename: str):
    """
    Save an Open3D PointCloud to a .txt file in COLMAP-style format:
    POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
    Dummy values of 1 are used for ERROR and TRACK[]
    """
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    if len(points) == 0:
        raise ValueError("Point cloud is empty")

    if colors.shape[0] != points.shape[0]:
        raise ValueError("Points and colors must have the same number of entries")

    colors_uint8 = (colors * 255).astype(np.uint8)

    with open(filename, 'w') as f:
        f.write("# 3D point list with one line of data per point: \n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points)}, mean track length: 1.0\n")

        for i, (pt, col) in enumerate(zip(points, colors_uint8), start=1):
            # Dummy values for error and a single (IMAGE_ID, POINT2D_IDX) pair
            error = 1.0
            image_id = 1
            point2d_idx = 1
            line = f"{i} {pt[0] + -1} {pt[1] + 1} {pt[2] + 2.5 } {col[0]} {col[1]} {col[2]} {error} {image_id} {point2d_idx}\n"
            f.write(line)



def write_camera_intrinsics_file(cameras, image_size, filename):
    """
    Write intrinsics of each camera to a COLMAP-style intrinsics file.

    Each camera is a dict with:
        - 'id': camera ID
        - 'intrinsics': 3x3 numpy array (K)
        - 'image_size': (width, height)

    Output file format:
    CAMERA_ID, MODEL, WIDTH, HEIGHT, fx, fy, cx, cy
    """
    with open(filename, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras)}\n")

        for cam in cameras:
            cam_id = cam["camera_id"]
            K = cam['intrinsics']
            width, height, params = extract_intrinsics_format(K, image_size)
            fx, fy, cx, cy = params
            f.write(f"{cam_id + 1} PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")


"""save_pcd_as_full_colmap_format_txt(pcd, "points3D.txt")
write_camera_intrinsics_file(cameras, image_size, "cameras.txt")"""


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


def get_extrinsics(image_id, extr_data, image_size):
    qvec, tvec = extract_extrinsics_format(extr_data)
    camera_id = image_id
    image_name = None
    xys = None
    point3D_ids = None

    camera_extr = BaseImage(
        id=image_id, qvec=qvec, tvec=tvec,
        camera_id=camera_id, name=image_name,
        xys=xys, point3D_ids=point3D_ids)

    return camera_extr

def normalize_quaternion(q):
    return q / np.linalg.norm(q)

def quaternion_angular_distance(q1, q2):
    q1 = normalize_quaternion(q1)
    q2 = normalize_quaternion(q2)
    dot = np.clip(np.dot(q1, q2), -1.0, 1.0)
    return 2 * np.arccos(np.abs(dot))  # Angle in radians

def match_and_save(exters_from_file, exters_from_gen, output_path="camera_matches.txt"):
    used_gen_ids = set()
    lines = []

    for ref_id, ref_cam in exters_from_file.items():
        ref_q = normalize_quaternion(ref_cam.qvec)
        ref_t = ref_cam.tvec

        best_match = None
        best_score = float('inf')

        for gen_id, gen_cam in exters_from_gen.items():
            if gen_id in used_gen_ids:
                continue

            gen_q = normalize_quaternion(gen_cam.qvec)
            gen_t = gen_cam.tvec

            t_dist = np.linalg.norm(ref_t - gen_t)
            r_dist = quaternion_angular_distance(ref_q, gen_q)
            score = t_dist + r_dist  # Simple combined metric

            if score < best_score:
                best_score = score
                best_match = (gen_id, gen_q, gen_t)

        if best_match:
            gen_id, gen_q, gen_t = best_match
            used_gen_ids.add(gen_id)

            line = f"{gen_id} {' '.join(map(str, gen_q))} {' '.join(map(str, gen_t))}    " \
                   f"{ref_id} {' '.join(map(str, ref_q))} {' '.join(map(str, ref_t))}    " \
                   f"{best_score:.6f}"
            lines.append(line)

    # Save to file
    with open(output_path, 'w') as f:
        header = "gen_id gen_qvec gen_tvec     ref_id ref_qvec ref_tvec     score"
        f.write(header + "\n")
        for line in lines:
            f.write(line + "\n")

    print(f"Saved {len(lines)} matched pairs to {output_path}")


def match_and_update(exters_from_file, exters_from_gen, output_path="camera_matches.txt"):
    used_gen_ids = set()
    lines = {}

    for ref_id, ref_cam in exters_from_file.items():
        ref_R = qvec2rotmat(ref_cam.qvec)
        ref_q = normalize_quaternion(rotmat2qvec(ref_R))
        ref_t = ref_cam.tvec

        best_match = None
        best_score = float('inf')

        for gen_id, gen_cam in exters_from_gen.items():
            if gen_id in used_gen_ids:
                continue

            gen_R = qvec2rotmat(gen_cam.qvec)
            gen_q = normalize_quaternion(rotmat2qvec(gen_R))
            gen_t = gen_cam.tvec

            t_dist = np.linalg.norm(ref_t - gen_t)
            r_dist = quaternion_angular_distance(ref_q, gen_q)
            score = t_dist + r_dist

            if score < best_score:
                best_score = score
                best_match = (gen_id, gen_q, gen_t)

        if best_match:
            gen_id, gen_q, gen_t = best_match
            used_gen_ids.add(gen_id)

            # Create new BaseImage using matched extrinsics
            exters_from_file[ref_id] = BaseImage(
                id=ref_id,
                qvec=gen_q,
                tvec=gen_t,
                camera_id=ref_cam.camera_id,
                name=ref_cam.name,
                xys=ref_cam.xys,
                point3D_ids=ref_cam.point3D_ids
            )

            line = f"{gen_id} {' '.join(map(str, gen_q))} {' '.join(map(str, gen_t))}    " \
                   f"{ref_id} {' '.join(map(str, ref_q))} {' '.join(map(str, ref_t))}    " \
                   f"{best_score:.6f}"
            lines[ref_id] = line

    with open(output_path, 'w') as f:
        f.write("gen_id gen_qvec gen_tvec     ref_id ref_qvec ref_tvec     score\n")
        for line in lines.values():
            f.write(line + "\n")

    print(f"Replaced {len(lines)} camera extrinsics in exters_from_file.")

exters_from_file = read_extrinsics_text('images.txt')


exters_from_gen = {}
for camera in cameras:
    extr_data = camera["extrinsics"]
    camera_id = camera["camera_id"]
    cam_extr = get_extrinsics(camera_id, extr_data, image_size)
    exters_from_gen[camera_id] = cam_extr

match_and_update(exters_from_file, exters_from_gen, output_path="matched_cameras.txt")


# frustums = []
# K = cameras[0]['intrinsics']  # Assuming all cameras share the same K
#
# for cam in exters_from_file.values():
#     R = qvec2rotmat(cam.qvec)
#     t = cam.tvec
#     extr = np.eye(4)
#     extr[:3, :3] = R
#     extr[:3, 3] = t
#     cam_frustum = create_camera_frustum(K, np.linalg.inv(extr),
#                                         image_size=(800, 800), scale=0.5, color=[1, 0, 0])
#     frustums.append(cam_frustum)
#
# o3d.visualization.draw_geometries([pcd] + frustums, window_name="Matched Cameras + Point Cloud")


inters_from_gen = {}
exters_from_gen = {}
for camera in generated_cameras:
    intr_data = camera["intrinsics"]
    extr_data = camera["extrinsics"]
    camera_id = camera["camera_id"]
    cam_pos = camera["position"]

    cam_intr = get_intrinsics(camera_id, intr_data, image_size)
    cam_extr = get_extrinsics(camera_id, extr_data, image_size)

    inters_from_gen[camera_id] = cam_intr
    exters_from_gen[camera_id] = cam_extr
