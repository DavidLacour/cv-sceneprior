import csv
import numpy as np
import cv2
import cv2 as cv
from ipdb import set_trace
import xml.etree.ElementTree as ET
import os
import pandas as pd
import sys

# Add the directory 'mymoodle' to Python path
sys.path.append(os.path.abspath('../'))

# Import from the gtm_hit module
from gtm_hit.misc.invision_calib import load_invision_calib
from gtm_hit.misc.geometry import (
    get_cuboid_from_ground_world, 
    get_projected_points, 
    get_bounding_box, 
    get_cuboid_from_ground_world2
)

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


def visualize_by_method(image_path, xml_path):
    if not os.path.exists(image_path) or not os.path.exists(xml_path):
        print(f"Error: Files not found")
        return
        
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get unique methods
    methods = set()
    for obj in root.findall('object'):
        method = obj.find('creation_method').text
        methods.add(method)
    
    # Create one image per method
    for method in methods:
        img = cv2.imread(image_path)
        
        for obj in root.findall('object'):
            current_method = obj.find('creation_method').text
            if current_method != method:
                continue
                
            # Get person_id
            person_id = obj.find('person_id').text
            
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            # Draw rectangle
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
            # Draw text inside rectangle
            text_y_offset = 25
            font_scale = 0.7
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Print full creation method on first line
            cv2.putText(img, f"Method: {current_method}", (xmin + 5, ymin + text_y_offset), 
                       font, font_scale, (0, 255, 0), 2)
            
            # Print person_id on second line
            cv2.putText(img, f"Person ID{current_method}: {person_id}", (xmin + 5, ymin + 2*text_y_offset), 
                       font, font_scale, (0, 255, 0), 2)

        # Display image with title showing the method
        window_name = f"Method: {method}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Configuration
    creation_method = "sync_SYNC17APR0908__sync_IVANA__existing_annotation"
    csv_file = "../../../annotationsdifferentsmethods.csv"
    params_dir = "../../../invisiondata/multicam-gt/annotation_dset/13apr/calibrations"
    output_folder = "../../../outputAnnotations2" + creation_method + "validatedandExisting"
    undistort = True

    # Test visualization
    image_path = '../../../invisiondata/multicam-gt/annotation_dset/13apr/frames/cam1/00003752.jpg'
    xml_path = output_folder + '/frame00003752_cam1_1.xml'
    visualize_by_method(image_path, xml_path)
