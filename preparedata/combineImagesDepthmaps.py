import os
import numpy as np
from PIL import Image

def add_depthmap_as_channel(image_path, depthmap_path, output_path):
    """
    Add a depth map as a fourth channel to an RGB image.

    Args:
    image_path (str): Path to the input RGB image file.
    depthmap_path (str): Path to the input depth map file (.npy).
    output_path (str): Path to save the resulting 4-channel image (.npy)
    """
    image = Image.open(image_path)
    image_array = np.array(image)
    depthmap = np.load(depthmap_path)
    
    if image_array.shape[:2] != depthmap.shape:
        raise ValueError(f"Image and depth map dimensions do not match for {image_path}")
    
    depthmap_reshaped = depthmap.reshape(*depthmap.shape, 1)
    
    combined = np.concatenate((image_array, depthmap_reshaped), axis=-1)
    
    np.save(output_path, combined)

def process_all_images(image_folder, depthmap_folder, output_folder):
    """
    Process all images in the given folder and combine them with their corresponding depth maps.

    Args:
    image_folder (str): Path to the folder containing input images.
    depthmap_folder (str): Path to the folder containing depth maps.
    output_folder (str): Path to the folder where combined images will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg'):
            parts = filename.split('_')
            if len(parts) < 5:
                print(f"Skipping file with unexpected format: {filename}")
                continue

            cam_num = parts[1]
            cam_subnum = parts[2]
            
            depthmap_filename = f"cam_{cam_num}_{cam_subnum}_calib_depth_map.npy"
            depthmap_path = os.path.join(depthmap_folder, depthmap_filename)
            
            if not os.path.exists(depthmap_path):
                print(f"Depth map not found for image: {filename}")
                continue
            
            image_path = os.path.join(image_folder, filename)
            output_filename = f"{os.path.splitext(filename)[0]}_combined.npy"
            output_path = os.path.join(output_folder, output_filename)
            
            try:
                add_depthmap_as_channel(image_path, depthmap_path, output_path)
                print(f"Processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

image_folder = "../../datatoprepare/train_data2/"
depthmap_folder = "../../datatoprepare/depthmaps/"
output_folder = "../../datatoprepare/combined_output_train_data2/"

process_all_images(image_folder, depthmap_folder, output_folder)
