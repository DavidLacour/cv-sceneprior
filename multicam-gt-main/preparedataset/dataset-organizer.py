import os
import shutil
from pathlib import Path
import re

annotationsPath = "../../../invisiondata/multicam-gt/annotation_dset/13apr/IvanAnnotations"
framesPath = "../../../invisiondata/multicam-gt/annotation_dset/13apr/frames"
base_dir = "../../../ivan5"

def create_directory_structure():
    """Create the required directory structure."""
    datasets = ["train_dataset", "val_dataset", "test_dataset"]
    subdirs = ["Annotations", "JPEGImages"]
    for dataset in datasets:
        for subdir in subdirs:
            Path(f"{base_dir}/{dataset}/{subdir}").mkdir(parents=True, exist_ok=True)

def get_camera_mapping():
    """Return the mapping of annotation suffixes to camera folders."""
    return {
        "1_1": "cam1",  # Train
        "1_2": "cam2",  # Test
        "1_3": "cam3",  # Train
        "1_4": "cam4",  # Train
        "2_1": "cam5",  # Val
        "2_2": "cam6",  # Train
        "2_3": "cam7",  # Train
        "2_4": "cam8"   # Train
    }

def get_dataset_mapping():
    """Return the mapping of camera suffixes to datasets."""
    return {
        "1_1": "train_dataset",
        "1_2": "test_dataset",
        "1_3": "train_dataset",
        "1_4": "train_dataset",
        "2_1": "val_dataset",
        "2_2": "train_dataset",
        "2_3": "train_dataset",
        "2_4": "train_dataset"
    }

def format_frame_number(frame_number):
    """Convert frame number to 8-digit format."""
    return str(int(frame_number)).zfill(8)

def get_new_filename(camera_suffix, frame_number, extension):
    """Generate new filename in the format cam1_1_distorted_00003094.xml"""
    formatted_frame = format_frame_number(frame_number)
    return f"cam_{camera_suffix}_distorted_{formatted_frame}{extension}"

def organize_dataset():
    """Main function to organize the dataset."""
    # Create directory structure
    create_directory_structure()
    
    # Get mappings
    camera_mapping = get_camera_mapping()
    dataset_mapping = get_dataset_mapping()
    
    # Process annotations
    annotations_dir = Path(annotationsPath)
    frames_dir = Path(framesPath)
    
    # Compile regex pattern for annotation files
    pattern = re.compile(r"frame(\d+)_cam(1_[1-4]|2_[1-4])\.xml")
    
    for annotation_file in annotations_dir.glob("*.xml"):
        match = pattern.match(annotation_file.name)
        if match:
            frame_number = match.group(1)
            camera_suffix = match.group(2)
            
            if camera_suffix in camera_mapping and camera_suffix in dataset_mapping:
                # Get corresponding image filename
                original_image_filename = f"{frame_number}.jpg"
                
                # Generate new filenames in the desired format
                new_annotation_filename = get_new_filename(camera_suffix, frame_number, ".xml")
                new_image_filename = get_new_filename(camera_suffix, frame_number, ".jpg")
                
                camera_folder = camera_mapping[camera_suffix]
                dataset_type = dataset_mapping[camera_suffix]
                
                # Source paths
                image_source = frames_dir / camera_folder / original_image_filename
                
                # Destination paths
                annotation_dest = Path(base_dir) / dataset_type / "Annotations" / new_annotation_filename
                image_dest = Path(base_dir) / dataset_type / "JPEGImages" / new_image_filename
                
                # Copy files if image exists
                if image_source.exists():
                    shutil.copy2(annotation_file, annotation_dest)
                    shutil.copy2(image_source, image_dest)
                else:
                    print(f"Warning: Image {image_source} not found for annotation {annotation_file.name}")

def main():
    try:
        organize_dataset()
        print("Dataset organization completed successfully!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()