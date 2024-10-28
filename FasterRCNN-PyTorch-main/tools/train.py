import torch
import argparse
import os
import numpy as np
import yaml
import random
from model.faster_rcnn import FasterRCNN
from tqdm import tqdm
from dataset.voc import VOCDataset
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import  Sampler
from torch.utils.tensorboard import SummaryWriter
import time
from datetime import datetime
import shutil
import zipfile
from tools.infer import evaluate_map

#not defined correctly on colab sometimes
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 


#num_samples_per_epoch = 1000  # Number of random images to use in each epoch

# Custom sampler for randomly selecting images in each epoch
class SubsetRandomSampler(Sampler):
    def __init__(self, dataset_size, num_samples):
        self.dataset_size = dataset_size
        self.num_samples = min(num_samples, dataset_size)

    def __iter__(self):
        # Randomly select a subset of indices for each epoch
        return iter(random.sample(range(self.dataset_size), self.num_samples))

    def __len__(self):
        return self.num_samples
    

def zip_logs(log_dir, output_path):
    """
    Create a zip file of the tensorboard logs
    """
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(log_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(log_dir))
                zipf.write(file_path, arcname)

def train(args):
    # Read the config file
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    if args.forcecpu:
        device = torch.device('cpu')
    
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']
    
    # Create unique run name with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"run_{timestamp}"
    
    # Initialize TensorBoard writer with specific run directory
    log_dir = os.path.join(train_config['task_name'], 'logs', run_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # Save config file in the log directory for reference
    config_save_path = os.path.join(log_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)
    
    writer = SummaryWriter(log_dir)
    
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    
    voc = VOCDataset('train',
                     im_dir=dataset_config['im_train_path'],
                     ann_dir=dataset_config['ann_train_path'])
    
    train_dataset = DataLoader(voc,
                             batch_size=1,
                             shuffle=True,
                             num_workers=2)
    
    faster_rcnn_model = FasterRCNN(model_config,
                                  num_classes=dataset_config['num_classes'])
    faster_rcnn_model.train()
    faster_rcnn_model.to(device)

    if not os.path.exists(train_config['task_name']):
        os.makedirs(train_config['task_name'])
    
    optimizer = torch.optim.SGD(lr=train_config['lr'],
                               params=filter(lambda p: p.requires_grad,
                                          faster_rcnn_model.parameters()),
                               weight_decay=5E-4,
                               momentum=0.9)
    scheduler = MultiStepLR(optimizer, milestones=train_config['lr_steps'], gamma=0.1)
    
    # Create a training info file
    train_info_path = os.path.join(log_dir, 'training_info.txt')
    with open(train_info_path, 'w') as f:
        f.write(f"Training started at: {timestamp}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Number of epochs: {train_config['num_epochs']}\n")
        f.write(f"Learning rate: {train_config['lr']}\n")
        f.write(f"Dataset size: {len(voc)}\n")
    
    acc_steps = train_config['acc_steps']
    num_epochs = train_config['num_epochs']
    step_count = 1
    global_step = 0
    training_start_time = time.time()

    try:
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            rpn_classification_losses = []
            rpn_localization_losses = []
            frcnn_classification_losses = []
            frcnn_localization_losses = []
            optimizer.zero_grad()
            
            for im, target, fname in tqdm(train_dataset):
                im = im.float().to(device)
                target['bboxes'] = target['bboxes'].float().to(device)
                target['labels'] = target['labels'].long().to(device)
                rpn_output, frcnn_output = faster_rcnn_model(im, target)
                
                rpn_loss = rpn_output['rpn_classification_loss'] + rpn_output['rpn_localization_loss']
                frcnn_loss = frcnn_output['frcnn_classification_loss'] + frcnn_output['frcnn_localization_loss']
                loss = rpn_loss + frcnn_loss
                
                # Log individual losses
                rpn_cls_loss = rpn_output['rpn_classification_loss'].item()
                rpn_loc_loss = rpn_output['rpn_localization_loss'].item()
                frcnn_cls_loss = frcnn_output['frcnn_classification_loss'].item()
                frcnn_loc_loss = frcnn_output['frcnn_localization_loss'].item()
                total_loss = loss.item()

                rpn_classification_losses.append(rpn_cls_loss)
                rpn_localization_losses.append(rpn_loc_loss)
                frcnn_classification_losses.append(frcnn_cls_loss)
                frcnn_localization_losses.append(frcnn_loc_loss)

                # Log to TensorBoard
                writer.add_scalar('Loss/Step/RPN_Classification', rpn_cls_loss, global_step)
                writer.add_scalar('Loss/Step/RPN_Localization', rpn_loc_loss, global_step)
                writer.add_scalar('Loss/Step/FRCNN_Classification', frcnn_cls_loss, global_step)
                writer.add_scalar('Loss/Step/FRCNN_Localization', frcnn_loc_loss, global_step)
                writer.add_scalar('Loss/Step/Total', total_loss, global_step)
                writer.add_scalar('Training/Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
        

                loss = loss / acc_steps
                loss.backward()
                if step_count % acc_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                step_count += 1
                global_step += 1

            epoch_time = time.time() - epoch_start_time
            print(f'Finished epoch {epoch}, time taken: {epoch_time:.2f}s')
            
            # Save model checkpoint
            checkpoint_path = os.path.join(log_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': faster_rcnn_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'global_step': global_step,
                'config': config
            }, checkpoint_path)

            # Log epoch metrics
            epoch_rpn_cls_loss = np.mean(rpn_classification_losses)
            epoch_rpn_loc_loss = np.mean(rpn_localization_losses)
            epoch_frcnn_cls_loss = np.mean(frcnn_classification_losses)
            epoch_frcnn_loc_loss = np.mean(frcnn_localization_losses)
            
            writer.add_scalar('Loss/Epoch/RPN_Classification', epoch_rpn_cls_loss, epoch)
            writer.add_scalar('Loss/Epoch/RPN_Localization', epoch_rpn_loc_loss, epoch)
            writer.add_scalar('Loss/Epoch/FRCNN_Classification', epoch_frcnn_cls_loss, epoch)
            writer.add_scalar('Loss/Epoch/FRCNN_Localization', epoch_frcnn_loc_loss, epoch)
            writer.add_scalar('Training/Epoch_Time', epoch_time, epoch)
            map = evaluate_map(args)
            writer.add_scalar('map',map,epoch)
            
            # Update training info file
            with open(train_info_path, 'a') as f:
                f.write(f"\nEpoch {epoch} completed in {epoch_time:.2f}s\n")
                f.write(f"Average losses:\n")
                f.write(f"  RPN Classification: {epoch_rpn_cls_loss:.4f}\n")
                f.write(f"  RPN Localization: {epoch_rpn_loc_loss:.4f}\n")
                f.write(f"  FRCNN Classification: {epoch_frcnn_cls_loss:.4f}\n")
                f.write(f"  FRCNN Localization: {epoch_frcnn_loc_loss:.4f}\n")
                f.write(f"  FRCNN map: {map:.4f}\n")
            scheduler.step()

    except Exception as e:
        print(f"Training interrupted: {str(e)}")
    finally:
        # Ensure we close the writer and save logs
        writer.close()
        
        # Calculate total training time
        total_time = time.time() - training_start_time
        with open(train_info_path, 'a') as f:
            f.write(f"\nTotal training time: {total_time:.2f}s\n")
        
        # Create zip file of logs
        logs_zip_path = os.path.join(train_config['task_name'], f'tensorboard_logs_{timestamp}.zip')
        zip_logs(log_dir, logs_zip_path)
        print(f"\nTensorBoard logs saved to: {logs_zip_path}")
        
        print("\nTo view these logs later:")
        print("1. Download the zip file from Colab")
        print("2. Extract it locally")
        print("3. Run: tensorboard --logdir=path_to_extracted_folder")
        print("4. Open http://localhost:6006 in your browser")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for faster rcnn training')
    parser.add_argument('--config', dest='config_path',
                        default='config/conf.yaml', type=str)
    parser.add_argument('--forcecpu', action='store_true',
                        help='Force using CPU even if CUDA is available')
    args = parser.parse_args()
    train(args)
