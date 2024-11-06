import numpy as np
import cv2 as cv
from collections import namedtuple
from enum import IntEnum
from django.conf import settings
from ipdb import set_trace
import point_cloud_utils as pcu

Calibration = namedtuple('Calibration', ['K', 'R', 'T', 'view_id'])
Bbox = namedtuple('Bbox', ['xc', 'yc', 'w', 'h'])  # , 'id', 'frame'])
Annotations = namedtuple(
    'Annotations', ['bbox', 'head', 'feet', 'height', 'id', 'frame', 'view'])
Homography = namedtuple('Homography', ['H', 'input_size', 'output_size'])

def get_simple_half_sphere():
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    # Define world plane size (for example, 100x100 units)
    world_plane_size = 100
    # Define the radius of the sphere (12.5% of world plane size for 25% coverage)
    radius = world_plane_size * 0.125
    # Define the number of latitude and longitude points
    num_pts = 100
    # Latitude and Longitude arrays (from 0 to pi for half-sphere and full circle respectively)
    latitude = np.linspace(0, np.pi / 2, num_pts)
    longitude = np.linspace(0, 2 * np.pi, num_pts)
    # Convert latitude and longitude to Cartesian coordinates
    x = radius * np.outer(np.sin(latitude), np.cos(longitude)) + 178360
    y = radius * np.outer(np.sin(latitude), np.sin(longitude)) + 211090
    z = 0.2 * radius * np.outer(np.cos(latitude), np.ones_like(longitude))
    # Compute faces
    faces = []
    for i in range(len(latitude) - 1):
        for j in range(len(longitude) - 1):
            # vertices of the first triangle
            v0 = (i * len(longitude) + j)
            v1 = (i * len(longitude) + (j + 1))
            v2 = ((i + 1) * len(longitude) + j)
            # vertices of the second triangle
            v3 = ((i + 1) * len(longitude) + (j + 1))
            # add the two triangles
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    faces = np.array(faces)
    v_sphere = np.array([x.flatten(), y.flatten(), z.flatten()]).T
    return v_sphere, faces

v,faces = get_simple_half_sphere() #todo: mesh here, sample sphere mesh is used for now


def reproject_to_world_ground_old(ground_pix, calib=None, undistort=False):
    """
    Compute world coordinate from pixel coordinate of point on the groundplane
    """
    # set_trace()
    undistort = settings.UNDISTORTED_FRAMES
    K0 = calib.K
    R0 = calib.R
    T0 = calib.T
    if undistort:
        K0 = calib.intrinsics.newCameraMatrix
        
        C0 = -R0.T @ T0
        l = R0.T @ calib.intrinsics.Rmat.T @ np.linalg.inv(K0) @ ground_pix
        world_point = C0 - l*(C0[2]/l[2])



    else: #distorted
        C0 = -R0.T @ T0
        l = R0.T @ np.linalg.inv(K0) @ ground_pix
        lambda1 = -1*(C0[2]/l[2])
        world_point = C0 + l*lambda1
        #world_point = world_point/world_point[2]
    
    return world_point

def reproject_to_world_ground(ground_pix, calib=None, undistort=False): #reproject to mesh
    """
    Compute world coordinate from pixel coordinate of point on the groundplane
    """
    #set_trace()
    undistort = settings.UNDISTORTED_FRAMES
    K0 = calib.K
    R0 = calib.R
    T0 = calib.T

    if undistort:
        K0 = calib.intrinsics.newCameraMatrix
        
        C0 = -R0.T @ T0 #cam pos in world coord
        ray_o = C0.reshape(1,-1)
        l = R0.T @ calib.intrinsics.Rmat.T @ np.linalg.inv(K0) @ ground_pix #camera centric ray
        
        ray_d = l.reshape(1,-1)
        ray_d /= np.linalg.norm(ray_d, axis=-1, keepdims=True)  # Normalize ray directions
        #world_point = C0 - l*(C0[2]/l[2])
        
        v,faces = get_simple_half_sphere()

        fid, bc, t = pcu.ray_mesh_intersection(v.astype(ray_o.dtype), faces, ray_o, ray_d)

        hit = fid.item() >= 0

        if not hit:
            world_point = C0 - l*(C0[2]/l[2])
            return world_point
        else:
            #set_trace()
            world_point = pcu.interpolate_barycentric_coords(faces, fid[fid>= 0], bc[fid>=0], v).reshape(-1,1)
            return world_point

    else: #distorted
        C0 = -R0.T @ T0
        l = R0.T @ np.linalg.inv(K0) @ ground_pix
        lambda1 = -1*(C0[2]/l[2])
        world_point = C0 + l*lambda1
        #world_point = world_point/world_point[2]
    
    return world_point

def move_with_mesh_intersection(ground_pix): #reproject to mesh
    ray_o = ground_pix.reshape(1,-1)
    ray_o[0,2] = -10
    ray_d = np.array([0,0,1],dtype=np.float).reshape(1,-1)
    fid, bc, t = pcu.ray_mesh_intersection(v.astype(ray_o.dtype), faces, ray_o, ray_d)
    hit = fid.item() >= 0
    
    if not hit:
        world_point = ground_pix
        world_point[2] = 0
        return world_point
    else:
        #set_trace()
        world_point = pcu.interpolate_barycentric_coords(faces, fid[fid>= 0], bc[fid>=0], v).reshape(-1,1)
        return world_point


def find_height_from_ground(obj, calib=None, undistort=False):
    x_bottom = (obj["x1"] + obj['x2'])/2
    y_bottom = obj['y2']
    ground_pix = np.array([[x_bottom], [y_bottom], [1]])
    world_point_bottom = reproject_to_world_ground(ground_pix, calib, undistort)

    y_top = obj['y1']
    top_pix = np.array([[x_bottom], [y_top], [1]])
    K0 = calib.K
    R0 = calib.R
    T0 = calib.T

    if undistort:
        K0 = calib.intrinsics.newCameraMatrix
        
        C0 = -R0.T @ T0
        l2 = R0.T @ calib.intrinsics.Rmat.T @ np.linalg.inv(K0) @ top_pix

        lambda2 = (world_point_bottom[0] - C0[0])/(l2[0])
        world_point_top = C0 + l2*lambda2
        return world_point_top

def project_world_to_camera(world_point, K1, R1, T1):
    """
    Project 3D point world coordinate to image plane (pixel coordinate)
    """
    point1 = ((R1 @ world_point) + T1)
    if (np.min(point1[2]) < 0):
        print("Projection of world point located behind the camera plane")
        return None, None
    point1 = K1 @ point1
    point1 = point1 / point1[2]

    return point1[:2]


def get_cuboid_from_ground_world2(world_point, calib, height, width, length, theta,undistort=False):
    CUBOID_VERTEX_COUNT = 10  # Assuming this is defined based on CuboidVertexEnum

    cuboid_points3d = np.zeros((CUBOID_VERTEX_COUNT, 3))
    
    # Adjust the coordinates to make world_point the lowest x, y, z
    cuboid_points3d[CuboidVertexEnum.FrontTopRight] = [width, length, height]
    cuboid_points3d[CuboidVertexEnum.FrontTopLeft] = [0, length, height]
    cuboid_points3d[CuboidVertexEnum.RearTopRight] = [width, 0, height]
    cuboid_points3d[CuboidVertexEnum.RearTopLeft] = [0, 0, height]
    cuboid_points3d[CuboidVertexEnum.FrontBottomRight] = [width, length, 0]
    cuboid_points3d[CuboidVertexEnum.FrontBottomLeft] = [0, length, 0]
    cuboid_points3d[CuboidVertexEnum.RearBottomRight] = [width, 0, 0]
    cuboid_points3d[CuboidVertexEnum.RearBottomLeft] = [0, 0, 0]
    cuboid_points3d[CuboidVertexEnum.Base] = [width/2, length/2, 0]
    cuboid_points3d[CuboidVertexEnum.Direction] = [width/2, length, 0]

    # Create rotation matrix
    rotz = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    # Apply rotation
    cuboid_points3d = (rotz @ cuboid_points3d.T).T

    # Apply translation
    cuboid_points3d = cuboid_points3d + world_point.T

    # Project 3D points to 2D
    cuboid_points2d = get_projected_points2(cuboid_points3d, calib,undistort)

    return cuboid_points2d

def get_cuboid_from_ground_world(world_point, calib, height, width, length,theta):
    
    cuboid_points3d = np.zeros((CUBOID_VERTEX_COUNT, 3))
    cuboid_points3d[CuboidVertexEnum.FrontTopRight] = [width / 2, length / 2, height]
    cuboid_points3d[CuboidVertexEnum.FrontTopLeft] = [-width / 2, length / 2, height]
    cuboid_points3d[CuboidVertexEnum.RearTopRight] = [width / 2, -length / 2, height]
    cuboid_points3d[CuboidVertexEnum.RearTopLeft] = [-width / 2, -length / 2, height]
    cuboid_points3d[CuboidVertexEnum.FrontBottomRight] = [width / 2, length / 2, 0]
    cuboid_points3d[CuboidVertexEnum.FrontBottomLeft] = [-width / 2, length / 2, 0]
    cuboid_points3d[CuboidVertexEnum.RearBottomRight] = [width / 2, -length / 2, 0]
    cuboid_points3d[CuboidVertexEnum.RearBottomLeft] = [-width / 2, -length / 2, 0]
    cuboid_points3d[CuboidVertexEnum.Base] = [0, 0, 0]
    cuboid_points3d[CuboidVertexEnum.Direction] = [0, length / 2, 0]
    #theta = np.pi/9 # 20 degree
    #set_trace()
    rotz = np.array([[np.cos(theta),-np.sin(theta),0],
                     [np.sin(theta), np.cos(theta),0],
                     [            0,             0,1]])
    cuboid_points3d = (rotz @ cuboid_points3d.T).T
    cuboid_points3d = cuboid_points3d + world_point.T
    #
    cuboid_points2d = get_projected_points(cuboid_points3d, calib)
    return cuboid_points2d
    print([{"x":x,"y":y} for x,y in cuboid_points2d])
    p1, p2 = get_bounding_box(cuboid_points2d)
    # x1, y1 = project_world_to_camera(top_left, calib.K, calib.R, calib.T)
    # x2, y2 = project_world_to_camera(bottom_right, calib.K, calib.R, calib.T)

    return *p1, *p2  # (x1, y1, x2+10, y2+10)

def get_cuboid2d_from_annotation(annotation, calib,undistort=False):
    #set_trace()
    height= annotation.object_size_x
    width = annotation.object_size_y
    length = annotation.object_size_z
    theta = annotation.rotation_theta
    world_point = annotation.world_point

    cuboid_points3d = np.zeros((CUBOID_VERTEX_COUNT, 3))
    cuboid_points3d[CuboidVertexEnum.FrontTopRight] = [width / 2, length / 2, height]
    cuboid_points3d[CuboidVertexEnum.FrontTopLeft] = [-width / 2, length / 2, height]
    cuboid_points3d[CuboidVertexEnum.RearTopRight] = [width / 2, -length / 2, height]
    cuboid_points3d[CuboidVertexEnum.RearTopLeft] = [-width / 2, -length / 2, height]
    cuboid_points3d[CuboidVertexEnum.FrontBottomRight] = [width / 2, length / 2, 0]
    cuboid_points3d[CuboidVertexEnum.FrontBottomLeft] = [-width / 2, length / 2, 0]
    cuboid_points3d[CuboidVertexEnum.RearBottomRight] = [width / 2, -length / 2, 0]
    cuboid_points3d[CuboidVertexEnum.RearBottomLeft] = [-width / 2, -length / 2, 0]
    cuboid_points3d[CuboidVertexEnum.Base] = [0, 0, 0]
    cuboid_points3d[CuboidVertexEnum.Direction] = [0, length / 2, 0]
    #theta = np.pi/9 # 20 degree
    #set_trace()
    rotz = np.array([[np.cos(theta),-np.sin(theta),0],
                     [np.sin(theta), np.cos(theta),0],
                     [            0,             0,1]])
    cuboid_points3d = (rotz @ cuboid_points3d.T).T
    cuboid_points3d = cuboid_points3d + world_point.T
    #
    cuboid_points2d = get_projected_points(cuboid_points3d,calib,undistort)
    return cuboid_points2d

class CuboidVertexEnum(IntEnum):
    FrontTopRight = 0
    FrontTopLeft = 1
    RearTopRight = 2
    RearTopLeft = 3
    FrontBottomRight = 4
    FrontBottomLeft = 5
    RearBottomRight = 6
    RearBottomLeft = 7
    Base = 8
    Direction=9
CUBOID_VERTEX_COUNT = 10

def get_projected_points2(points3d, calib, undistort=False):
    points3d = np.array(points3d).reshape(-1, 3)
    Rvec = calib.extrinsics.get_R_vec()  # cv.Rodrigues
    Tvec = calib.extrinsics.T
    points2d, _ = cv.projectPoints(
        points3d, Rvec, Tvec, calib.intrinsics.cameraMatrix, calib.intrinsics.distCoeffs)
    if undistort:
        #set_trace()
        points3d_cam = calib.extrinsics.R @ points3d.T + calib.extrinsics.T.reshape(-1,1)
        in_front_of_camera = (points3d_cam[2, :] > 0).all()
        if not in_front_of_camera:
            raise ValueError("Points are not in camera view.")
        points3d_cam_rectified = calib.intrinsics.Rmat @ points3d_cam #correct the slant of the camera
        points2d = calib.intrinsics.newCameraMatrix @ points3d_cam_rectified

        points2d = points2d[:2,:]/points2d[2,:]
        points2d = points2d.T
    points2d = np.squeeze(points2d)
    points2d = [tuple(p) for p in points2d]
    return points2d

def get_projected_points(points3d, calib, undistort=False):
    undistort=False
    #undistort = settings.UNDISTORTED_FRAMES
    points3d = np.array(points3d).reshape(-1, 3)
    Rvec = calib.extrinsics.get_R_vec()  # cv.Rodrigues
    Tvec = calib.extrinsics.T
    points2d, _ = cv.projectPoints(
        points3d, Rvec, Tvec, calib.intrinsics.cameraMatrix, calib.intrinsics.distCoeffs)
    if undistort:
        #set_trace()
        points3d_cam = calib.extrinsics.R @ points3d.T + calib.extrinsics.T.reshape(-1,1)
        in_front_of_camera = (points3d_cam[2, :] > 0).all()
        if not in_front_of_camera:
            raise ValueError("Points are not in camera view.")
        points3d_cam_rectified = calib.intrinsics.Rmat @ points3d_cam #correct the slant of the camera
        points2d = calib.intrinsics.newCameraMatrix @ points3d_cam_rectified

        points2d = points2d[:2,:]/points2d[2,:]
        points2d = points2d.T
    points2d = np.squeeze(points2d)
    points2d = [tuple(p) for p in points2d]
    return points2d


def triangulate_point(points_2d, multi_calib):
    # Need at least point of view
    assert points_2d.shape[0] > 1

    # compute camera position for each view
    camera_positions = [-calib.R.T @ calib.T for calib in multi_calib]

    # Compute 3D direction from camera toward point
    point_directions = [-calib.R.T @ np.linalg.inv(
        calib.K) @ point for point, calib in zip(points_2d, multi_calib)]

    point_3d = nearest_intersection(
        np.array(camera_positions).squeeze(2), np.array(point_directions))

    return point_3d


def nearest_intersection(points, dirs):
    """
    :param points: (N, 3) array of points on the lines
    :param dirs: (N, 3) array of unit direction vectors
    :returns: (3,) array of intersection point

    from https://stackoverflow.com/questions/52088966/nearest-intersection-point-to-many-lines-in-python
    """
    # normalized direction
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs_mat = dirs[:, :, np.newaxis] @ dirs[:, np.newaxis, :]
    points_mat = points[:, :, np.newaxis]
    I = np.eye(3)
    return np.linalg.lstsq(
        (I - dirs_mat).sum(axis=0),
        ((I - dirs_mat) @ points_mat).sum(axis=0),
        rcond=None
    )[0]


def project_roi_world_to_camera(world_point, K1, R1, T1):
    """
    Project Region of interest 3D point world coordinate to image plane (pixel coordinate)
    A bit Hacky since world coordinate are sometime behind image plane, we interpolate between corner of polygon
    to only keep point in front of the image plane
    """

    point1 = ((R1 @ world_point) + T1)

    if point1[2].min() < 0:
        # If a corner point of the roi lie behind the image compute corespondence in the image plane
        x = world_point[0]
        y = world_point[1]

        # Evenly sample point around polygon define by corner point in world_point
        distance = np.cumsum(
            np.sqrt(np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2))
        distance = distance/distance[-1]

        fx, fy = interp1d(distance, x), interp1d(distance, y)

        alpha = np.linspace(0, 1, 150)
        x_regular, y_regular = fx(alpha), fy(alpha)

        world_point = np.vstack(
            [x_regular, y_regular, np.zeros(x_regular.shape)])

        point1 = ((R1 @ world_point) + T1)

        # Filter out point behind the camera plane (Z < 0)
        point1 = np.delete(point1, point1[2] < 0, axis=1)
    point1 = K1 @ point1
    point1 = point1 / point1[2]

    return point1[:2]


def update_img_point_boundary(img_points, view_ground_edge):
    # Make sure that all the img point are inside the image, if there are not replace them by points on the boundary
    img_points = map(Point, img_points)
    # img_corners = map(Point, [(0.0, 0.0), (0.0, img_size[0]), (img_size[1], img_size[0]), (img_size[1], 0.0)])
    img_corners = map(Point, view_ground_edge)

    poly1 = Polygon(*img_points)
    poly2 = Polygon(*img_corners)
    isIntersection = intersection(poly1, poly2)  # poly1.intersection(poly2)

    point_inside = list(isIntersection)
    point_inside.extend([p for p in poly1.vertices if poly2.encloses_point(p)])
    point_inside.extend([p for p in poly2.vertices if poly1.encloses_point(p)])

    boundary_updated = convex_hull(*point_inside).vertices
    boundary_updated = [p.coordinates for p in boundary_updated]

    return np.stack(boundary_updated).astype(float)


def update_K_after_resize(K, old_size, new_size):
    fx = 1.0 / (old_size[1] / new_size[1])
    fy = 1.0 / (old_size[0] / new_size[0])

    scaler = np.array([
        [fx, 0, 0],
        [0, fy, 0],
        [0, 0, 1]]
    )

    new_K = scaler @ K

    return new_K


def rescale_keypoints(points, org_img_dim, out_img_dim):

    if len(points) == 0:
        return points

    out_img_dim = np.array(out_img_dim)
    org_img_dim = np.array(org_img_dim)

    if np.all(org_img_dim == out_img_dim):
        return points

    resize_factor = out_img_dim / org_img_dim
    # swap x and y
    resize_factor = resize_factor[::-1]

    resized_points = points*resize_factor

    return resized_points


def distance_point_to_line(lp1, lp2, p3):
    # Both point of the line are the same return distance to that point
    if np.all(lp1 == lp2):
        return np.linalg.norm(p3-lp1)
    else:
        return np.abs(np.cross(lp2-lp1, lp1-p3) / np.linalg.norm(lp2-lp1))


def get_bounding_box(points):
    points = points.copy()
    #set_trace()
    try:
        points = np.array(points, dtype=np.int64).reshape(-1, 1, 2)
        x, y, w, h = cv.boundingRect(points)
    except OverflowError:
        return points[0], points[4]
    return  (x, y), (x+w, y+h)
