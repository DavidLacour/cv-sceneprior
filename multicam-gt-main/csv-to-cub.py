import csv
import numpy as np
import cv2 as cv
from ipdb import set_trace
import xml.etree.ElementTree as ET
import pandas as pd 

# Assuming these are defined elsewhere in your code
from gtm_hit.misc.invision_calib import load_invision_calib
from gtm_hit.misc.geometry import get_cuboid_from_ground_world, get_projected_points, get_bounding_box

# Modified get_projected_points function
"""
def get_projected_points(points3d, calib, undistort=True):
    points3d = np.array(points3d).reshape(-1, 3)
    Rvec = calib.extrinsics.get_R_vec()  # cv.Rodrigues
    Tvec = calib.extrinsics.T
    points2d, _ = cv.projectPoints(
        points3d, Rvec, Tvec, calib.intrinsics.cameraMatrix, calib.intrinsics.distCoeffs)
    if undistort:
        points3d_cam = calib.extrinsics.R @ points3d.T + calib.extrinsics.T.reshape(-1,1)
        in_front_of_camera = (points3d_cam[2, :] > 0).all()
        if not in_front_of_camera:
            raise ValueError("Points are not in camera view.")
        points3d_cam_rectified = calib.intrinsics.Rmat @ points3d_cam  # correct the slant of the camera
        points2d = calib.intrinsics.newCameraMatrix @ points3d_cam_rectified
        points2d = points2d[:2,:]/points2d[2,:]
        points2d = points2d.T
    points2d = np.squeeze(points2d)
    points2d = [tuple(p) for p in points2d]
    return points2d

def get_cuboid_from_ground_world(world_point, calib, height, width, length, theta, undistort=True):
    CUBOID_VERTEX_COUNT = 10  # Assuming this is defined somewhere

    cuboid_points3d = np.zeros((CUBOID_VERTEX_COUNT, 3))
    cuboid_points3d[0] = [width / 2, length / 2, height]  # FrontTopRight
    cuboid_points3d[1] = [-width / 2, length / 2, height]  # FrontTopLeft
    cuboid_points3d[2] = [width / 2, -length / 2, height]  # RearTopRight
    cuboid_points3d[3] = [-width / 2, -length / 2, height]  # RearTopLeft
    cuboid_points3d[4] = [width / 2, length / 2, 0]  # FrontBottomRight
    cuboid_points3d[5] = [-width / 2, length / 2, 0]  # FrontBottomLeft
    cuboid_points3d[6] = [width / 2, -length / 2, 0]  # RearBottomRight
    cuboid_points3d[7] = [-width / 2, -length / 2, 0]  # RearBottomLeft
    cuboid_points3d[8] = [0, 0, 0]  # Base
    cuboid_points3d[9] = [0, length / 2, 0]  # Direction

    rotz = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    cuboid_points3d = (rotz @ cuboid_points3d.T).T
    cuboid_points3d = cuboid_points3d + world_point.T

    cuboid_points2d = get_projected_points(cuboid_points3d, calib, undistort)
    return cuboid_points2d


def load_csv_and_generate_cuboids(csv_file, params_dir, undistort=True):
    # Load camera parameters
    cam_params = load_invision_calib(params_dir)

    cuboids_2d = {}

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            frame_id = int(row['frame_id'])
            person_id = int(row['person_id'])
            
            world_point = np.array([float(row['Xw']), float(row['Yw']), float(row['Zw'])]).reshape(-1, 1)
            height = float(row['object_size_x'])
            width = float(row['object_size_y'])
            length = float(row['object_size_z'])
            theta = float(row['rotation_theta'])

            annotation_key = f"{frame_id}_{person_id}"
            cuboids_2d[annotation_key] = {}

            for cam_idx, calib in cam_params.items():
                cuboid_2d = get_cuboid_from_ground_world(world_point, calib, height, width, length, theta, undistort)
                cuboids_2d[annotation_key][cam_idx] = cuboid_2d

    return cuboids_2d
"""    


def is_point_in_image(point, min_x, max_x, min_y, max_y):
    return min_x <= point[0] <= max_x and min_y <= point[1] <= max_y

def clamp_coordinates(coord, min_x, max_x, min_y, max_y):
    return [
        max(min_x, min(max_x, coord[0])),
        max(min_y, min(max_y, coord[1]))
    ]

def get_bounding_box(points):
    x_coords, y_coords = zip(*points)
    return (min(x_coords), min(y_coords)), (max(x_coords), max(y_coords))
"""
def load_csv_and_generate_xml(csv_file, params_dir, output_xml, undistort=True):
    cam_params = load_invision_calib(params_dir)
    cuboids_2d = {}

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            frame_id = int(row['frame_id'])
            person_id = int(row['person_id'])
            world_point = np.array([float(row['Xw']), float(row['Yw']), float(row['Zw'])]).reshape(-1, 1)
            height = float(row['object_size_x'])
            width = float(row['object_size_y'])
            length = float(row['object_size_z'])
            theta = float(row['rotation_theta'])

            if frame_id not in cuboids_2d:
                cuboids_2d[frame_id] = {}

            for cam_idx, calib in cam_params.items():
                if cam_idx not in cuboids_2d[frame_id]:
                    cuboids_2d[frame_id][cam_idx] = []

                cuboid_2d = get_cuboid_from_ground_world(world_point, calib, height, width, length, theta)
                


                # Check if at least one point is in the image
                if any(is_point_in_image(point, 0, 1920, 0, 1080) for point in cuboid_2d):
                    clamped_cuboid = [clamp_coordinates(point, 0, 1920, 0, 1080) for point in cuboid_2d]
                    cuboids_2d[frame_id][cam_idx].append((person_id, clamped_cuboid))
"""
def load_csv_and_generate_xml(csv_file, params_dir, output_xml, undistort=True):
    cam_params = load_invision_calib(params_dir)
    cuboids_2d = {}

    # Generate XML for the first frame of the first camera
    first_frame = min(cuboids_2d.keys())
    first_camera = min(cuboids_2d[first_frame].keys())

    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "VOC2007"

    camera_id = cam_params[first_camera].id
    ET.SubElement(root, "filename").text = f"cam_{camera_id[0]}_{camera_id[1]}_undistorted_{first_frame:08d}.jpg"

    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "The VOC2007 Database"
    ET.SubElement(source, "annotation").text = "PASCAL VOC2007"
    ET.SubElement(source, "image").text = "flickr"

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = "1920"
    ET.SubElement(size, "height").text = "1080"
    ET.SubElement(size, "depth").text = "3"

    ET.SubElement(root, "segmented").text = "0"

    for person_id, cuboid in cuboids_2d[first_frame][first_camera]:
        (xmin, ymin), (xmax, ymax) = get_bounding_box(cuboid)
        
        # Additional check to ensure the bounding box is valid
        if xmin < xmax and ymin < ymax:
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = "person"
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"

            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(int(xmin))
            ET.SubElement(bndbox, "ymin").text = str(int(ymin))
            ET.SubElement(bndbox, "xmax").text = str(int(xmax))
            ET.SubElement(bndbox, "ymax").text = str(int(ymax))

            ET.SubElement(obj, "confidence").text = "1.0"  # Placeholder confidence

    tree = ET.ElementTree(root)
    tree.write(output_xml)

# Usage
csv_file = "../../gtm_hit_annotations.csv"
params_dir = "../../invisiondata/multicam-gt/annotation_dset/13apr/calibrations"
output_xml = "annotations.xml"
undistort = True  # Set this to True if you want undistorted projections

load_csv_and_generate_xml(csv_file, params_dir, output_xml, undistort)
print(f"XML file '{output_xml}' has been generated.")