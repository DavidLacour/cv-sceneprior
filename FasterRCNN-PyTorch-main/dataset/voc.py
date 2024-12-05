import glob
import os
import random
import cv2
import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
import xml.etree.ElementTree as ET
import numpy as np
import re

def load_images_and_anns(im_dir, ann_dir,depth_dir, label2idx):
    r"""
    Method to get the xml files and for each file
    get all the objects and their ground truth detection
    information for the dataset
    :param im_dir: Path of the images
    :param ann_dir: Path of annotation xmlfiles
    :depth_dir: Path of depth maps 
    :param label2idx: Class Name to index mapping for dataset
    :return:
    """
    im_infos = []
    for ann_file in tqdm(glob.glob(os.path.join(ann_dir, '*.xml'))):
        im_info = {}
        #im_info['img_id'] =  "cam_1_3_undistorted_00000004" #os.path.basename(ann_file).split('.xml')[0]
        im_info['img_id'] = os.path.basename(ann_file).split('.xml')[0]
        im_info['filename'] = os.path.join(im_dir, '{}.jpg'.format(im_info['img_id']))
        camera_name = '_'.join(im_info['img_id'].split('_')[:3])
        depth_filename = f"{camera_name}_depth.npy"
        depth_path = os.path.join(depth_dir, depth_filename)
        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"Depth file not found: {depth_path}")
        
                 
        camera_name = '_'.join(im_info['img_id'].split('_')[:3])
        depth_filename = f"{camera_name}_depth.npy"
        depth_path = os.path.join(depth_dir, depth_filename)
        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"Depth file not found: {depth_path}")

        
       

        im_info['depthfilename'] = depth_path

        
        ann_info = ET.parse(ann_file)
        root = ann_info.getroot()
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        im_info['width'] = width
        im_info['height'] = height
        detections = []
        my_objs = ann_info.findall('object')
        if(len(my_objs)) == 0:
            "NO OBJECT 77777777777777777777777777777777777777777777777777777777777777777777777777777"
        for obj in ann_info.findall('object'):
            error_details = []
            # Check for name element
            name_elem = obj.find('name')
            if name_elem is None:
                error_details.append("Missing 'name' element")
                print("name is none ")
            elif not name_elem.text:
                error_details.append("Empty 'name' element")
                print("Empty 'name' element")
            if name_elem.text != "person":
                print("NOT A PERSON 777777777777777777777777"+ name_elem.text)
          
            # Check for bndbox element and its children
            bbox_elem = obj.find('bndbox')
            if bbox_elem is None:
                error_details.append("Missing 'bndbox' element")
                print("Missing 'bndbox' element")
            else:
                required_bbox_fields = ['xmin', 'ymin', 'xmax', 'ymax']
                for field in required_bbox_fields:
                    field_elem = bbox_elem.find(field)
                    if field_elem is None:
                        error_details.append(f"Missing '{field}' in bndbox")
                        print("Missing field ")
                    elif not field_elem.text:
                        error_details.append(f"Empty '{field}' value in bndbox")
                        print("empty value")
                    else:
                        try:
                            # Verify bbox values can be converted to float
                            float(field_elem.text)
                        except ValueError:
                            error_details.append(f"Invalid number format for '{field}': {field_elem.text}")
            if error_details:
                print(error_details)
            
            det = {}
            label = label2idx[obj.find('name').text]
            bbox_info = obj.find('bndbox')
            bbox = [
                int(float(bbox_info.find('xmin').text))-1,
                int(float(bbox_info.find('ymin').text))-1,
                int(float(bbox_info.find('xmax').text))-1,
                int(float(bbox_info.find('ymax').text))-1
            ]
            det['label'] = label
            det['bbox'] = bbox
            detections.append(det)
            
        im_info['detections'] = detections
        im_infos.append(im_info)
    print('Total {} images found'.format(len(im_infos)))
    return im_infos


def normalize_depth_map(depth_map):
    """
    Normalize depth map using robust statistics to handle outliers
    
    Args:
        depth_map (np.ndarray): Raw depth map
        
    Returns:
        np.ndarray: Normalized depth map between 0 and 1
    """
    # Convert to float32 for processing
    depth = depth_map.astype(np.float32)
    
    # Remove invalid values (if any)
    valid_mask = depth > 0
    if valid_mask.sum() == 0:
        return np.zeros_like(depth)
    
    valid_depth = depth[valid_mask]
    
    # Calculate robust statistics
    depth_min = np.percentile(valid_depth, 2)  # 2nd percentile instead of minimum
    depth_max = np.percentile(valid_depth, 98)  # 98th percentile instead of maximum
    
    # Clip depth values to remove outliers
    depth = np.clip(depth, depth_min, depth_max)
    
    # Normalize to [0, 1]
    depth = (depth - depth_min) / (depth_max - depth_min + 1e-8)
    
    return depth

class VOCDataset(Dataset):
    def __init__(self, split, im_dir, ann_dir,depth_dir):
        self.split = split
        self.im_dir = im_dir
        self.ann_dir = ann_dir
        classes = [
            'person'
        ]
        classes = sorted(classes)
        classes = ['background'] + classes
        self.label2idx = {classes[idx]: idx for idx in range(len(classes))}
        self.idx2label = {idx: classes[idx] for idx in range(len(classes))}
        print(self.idx2label)
        self.images_info = load_images_and_anns(im_dir, ann_dir, depth_dir, self.label2idx)
    
    def __len__(self):
        return len(self.images_info)
    
    def load_image_with_depth(self, image_path, depth_path, normalize=True):
        """
        Load and combine RGB image with depth map
        
        Args:
            image_path (str): Path to RGB image
            depth_path (str): Path to depth map .npy file
            normalize (bool): Whether to normalize the depth map
            
        Returns:
            torch.Tensor: Combined RGBD tensor with shape (4, H, W)
        """
        rgb_image = Image.open(image_path)
        rgb_tensor = torchvision.transforms.ToTensor()(rgb_image)
        
        # Load depth map
        depth_map = np.load(depth_path)
        
        if normalize:
            depth_map = normalize_depth_map(depth_map)
        
        depth_tensor = torch.from_numpy(depth_map).float().unsqueeze(0)
        
        if depth_tensor.shape[1:] != rgb_tensor.shape[1:]:
            raise ValueError(f"Dimension mismatch: RGB shape {rgb_tensor.shape}, Depth shape {depth_tensor.shape}")
        
        rgbd_tensor = torch.cat([rgb_tensor, depth_tensor], dim=0)
        return rgbd_tensor, rgb_image.size
    
    def flip_rgbd_image(self, rgbd_tensor):
        """
        Flip RGBD image horizontally
        
        Args:
            rgbd_tensor (torch.Tensor): RGBD tensor with shape (4, H, W)
            
        Returns:
            torch.Tensor: Flipped RGBD tensor
        """
        return torch.flip(rgbd_tensor, dims=[2])
    
    def __getitem__(self, index):
       
        im_info = self.images_info[index]
        im = Image.open(im_info['filename'])
        #im = cv2.imread(im_info['filename'])
        #im = Image.open(im_info['filename'])

        try:
            rgbd_tensor, image_size = self.load_image_with_depth(
                im_info['filename'],
                im_info['depthfilename']
            )
        except Exception as e:
            print(f"Error loading images werhzwr4thwh: {e}")
            print(f"Image path: {im_info['filename']}")
            print(f"Depth path: {im_info['depthfilename']}")
            raise e

        #im = np.load(im_info['filename'])
        to_flip = False
        if self.split == 'train' and random.random() < 0.5:
            to_flip = True
            rgbd_tensor = self.flip_rgbd_image(rgbd_tensor)

        im_tensor = torchvision.transforms.ToTensor()(im)
        targets = {}
        targets['bboxes'] = torch.as_tensor([detection['bbox'] for detection in im_info['detections']])
        targets['labels'] = torch.as_tensor([detection['label'] for detection in im_info['detections']])
        if to_flip:
            for idx, box in enumerate(targets['bboxes']):
                x1, y1, x2, y2 = box
                w = x2-x1
                im_w = im_tensor.shape[-1]
                x1 = im_w - x1 - w
                x2 = x1 + w
                targets['bboxes'][idx] = torch.as_tensor([x1, y1, x2, y2])
        return rgbd_tensor, targets, im_info['filename']
        