import csv
import numpy as np
import cv2 as cv
from ipdb import set_trace
import xml.etree.ElementTree as ET
import os
import pandas as pd 
from gtm_hit.misc.invision_calib import load_invision_calib
from gtm_hit.misc.geometry import get_cuboid_from_ground_world, get_projected_points, get_bounding_box, get_cuboid_from_ground_world2




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


def load_csv_and_generate_xml(csv_file, params_dir, output_folder,creation_method ,validated=True, undistort=True):
    os.makedirs(output_folder, exist_ok=True)
    
    cam_params = load_invision_calib(params_dir)

    df = pd.read_csv(csv_file)
    
   
    df['frame_id'] = df['frame_id'].astype(int)
    df['person_id'] = df['person_id'].astype(int)
    
 
    df = df.sort_values('frame_id')
    unique_names = df[~df["creation_method"].str.contains("imported", case=False, na=False)]["creation_method"].unique()
    for name in unique_names:
        print(name)

    if validated:
        df = df[df["validated"] == "t"]

    if creation_method == "imported":
        df = df[df["creation_method"].str.contains("imported")]
    else: 
       df = df[
    (df["creation_method"].str.contains("imported")) | 
    (df["creation_method"] == creation_method)
]

    for frame_id, frame_data in df.groupby('frame_id', sort=True):
        #frame_id = frame_id + 3150
        cuboids_2d = {cam_idx: [] for cam_idx in cam_params.keys()}

        for _, row in frame_data.iterrows():

            world_point = np.array([row['Xw'], row['Yw'], row['Zw'] ]).reshape(-1, 1)
            width = row['object_size_y']
            height = row['object_size_x']
            length = row['object_size_z']
            theta = row['rotation_theta']

           
            for cam_idx, calib in cam_params.items():
                cuboid_2d = get_cuboid_from_ground_world(world_point, calib, height, width, length, theta)
                


                # Check if at least one point is in the image and clamp if necessary
                if any(is_point_in_image(point, 0, 1920, 0, 1080) for point in cuboid_2d):
                    clamped_cuboid = [clamp_coordinates(point, 0, 1920, 0, 1080) for point in cuboid_2d]
                    cuboids_2d[cam_idx].append((row['person_id'], clamped_cuboid))

        # Generate XML for each camera
        for cam_idx, cuboids in cuboids_2d.items():
            if not cuboids:
                continue  

            root = ET.Element("annotation")
            ET.SubElement(root, "folder").text = "VOC2007"

            camera_id = cam_params[cam_idx].id
            ET.SubElement(root, "filename").text = f"cam_{camera_id[0]}_{camera_id[1]}_undistorted_{frame_id:08d}.jpg"

            source = ET.SubElement(root, "source")
            ET.SubElement(source, "database").text = "The VOC2007 Database"
            ET.SubElement(source, "annotation").text = "PASCAL VOC2007"
            ET.SubElement(source, "image").text = "flickr"

            size = ET.SubElement(root, "size")
            ET.SubElement(size, "width").text = "1920"
            ET.SubElement(size, "height").text = "1080"
            ET.SubElement(size, "depth").text = "3"

            ET.SubElement(root, "segmented").text = "0"

            for person_id, cuboid in cuboids:
                (xmin, ymin), (xmax, ymax) = get_bounding_box(cuboid)
                
                
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

                    ET.SubElement(obj, "confidence").text = "1.0"  

            tree = ET.ElementTree(root)
            output_xml = os.path.join(output_folder, f"frame{frame_id:08d}_cam{camera_id[0]}_{camera_id[1]}.xml")
            tree.write(output_xml)
            print(f"Generated XML file: {output_xml}")
# imported means the name contains imported 
"""
imported
existing_annotation
sync_IVANA__existing_annotation
sync_SYNC17APR0908__sync_IVANA__existing_annotation
sync_ANA__existing_annotation
sync_SYNC17APR0908__sync_ANA__existing_annotation
"""

creation_method = "sync_SYNC17APR0908__sync_IVANA__existing_annotation"
csv_file = "../../annotationsdifferentsmethods.csv"
params_dir = "../../invisiondata/multicam-gt/annotation_dset/13apr/calibrations"
output_folder = "../../outputAnnotations2" + creation_method + "validatedandExisting"
undistort = True  #might not work  

load_csv_and_generate_xml(csv_file, params_dir, output_folder, creation_method,undistort)
print("XML files generation completed.")

import os
import cv2
import xml.etree.ElementTree as ET

def draw_annotations(image_path, xml_path):
    
    img = cv2.imread(image_path)

   
    tree = ET.parse(xml_path)
    root = tree.getroot()

    
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        cv2.putText(img, name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Image', img)
    cv2.waitKey(0)  
   
# 00003752
# 00003192


image_path = '../../invisiondata/multicam-gt/annotation_dset/13apr/frames/cam1/00003752.jpg'
xml_path =  output_folder + '/frame00003752_cam1_1.xml'
#xml_path =  output_folder + '/frame00003192_cam1_1.xml'

if not os.path.exists(image_path):
    print(f"Error: Image file not found: {image_path}")
    

if not os.path.exists(xml_path):
    print(f"Error: XML file not found: {xml_path}")
    

draw_annotations(image_path, xml_path)
cv2.destroyAllWindows()