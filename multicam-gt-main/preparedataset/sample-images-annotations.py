import os
import random
import shutil
from pathlib import Path
from google.colab import drive

# Mount Google Drive before
#drive.mount('/content/drive')

def create_sample_folders(base_path, annotation_dir, images_dir, sample_size=1000, random_seed=42):
    """
    Creates sample folders with paired images and annotations using a fixed seed.
    
    Args:
        base_path: Path to Google Drive working directory
        annotation_dir: Name of annotations folder
        images_dir: Name of images folder
        sample_size: Number of samples to create
        random_seed: Seed for reproducible sampling
    """
  
    random.seed(random_seed)
    
    
    annotations_path = Path(base_path) / annotation_dir
    images_path = Path(base_path) / images_dir
    
    new_annotations_dir = Path(base_path) / f"{annotation_dir}1000"
    new_images_dir = Path(base_path) / f"{images_dir}1000"
    
    # Remove existing directories if they exist
    if new_annotations_dir.exists():
        shutil.rmtree(new_annotations_dir)
    if new_images_dir.exists():
        shutil.rmtree(new_images_dir)
    
    # Create fresh directories
    new_annotations_dir.mkdir(exist_ok=True)
    new_images_dir.mkdir(exist_ok=True)
    
   
    xml_files = list(annotations_path.glob("*.xml"))
    
    
    pairs = {}
    for xml_file in xml_files:
        jpg_name = xml_file.stem + ".jpg"
        jpg_file = images_path / jpg_name
        
        if jpg_file.exists():
            pairs[xml_file] = jpg_file
    
    
    if len(pairs) < sample_size:
        print(f"Warning: Only {len(pairs)} valid pairs found, using all of them")
        sample_size = len(pairs)
    
    sampled_pairs = random.sample(list(pairs.items()), sample_size)
    
   
    for xml_file, jpg_file in sampled_pairs:
        shutil.copy2(xml_file, new_annotations_dir / xml_file.name)
        shutil.copy2(jpg_file, new_images_dir / jpg_file.name)
    
    
    with open(Path(base_path) / "sampled_files.txt", "w") as f:
        f.write(f"Random seed: {random_seed}\n")
        f.write(f"Total samples: {sample_size}\n\n")
        f.write("Sampled pairs:\n")
        for xml_file, jpg_file in sampled_pairs:
            f.write(f"{xml_file.name} -> {jpg_file.name}\n")
        
    print(f"Successfully copied {sample_size} pairs to new directories")
    print(f"New annotations directory: {new_annotations_dir}")
    print(f"New images directory: {new_images_dir}")
    print(f"List of sampled files saved to: {Path(base_path) / 'sampled_files.txt'}")
    print(f"Used random seed: {random_seed}")

# PATH AND DIR
base_path = "/content/drive/MyDrive/CV/cv/cloe"  
create_sample_folders(
    base_path=base_path,
    annotation_dir="train_dataset/Annotations",
    images_dir="train_dataset/JPEGImages",
    sample_size=1000,
    random_seed=42 
)
