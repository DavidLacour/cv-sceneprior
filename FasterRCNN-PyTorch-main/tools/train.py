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
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import shutil
import zipfile
import time
from tools.infer import evaluate_map

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# predefined collate doesn't work with different size 
def custom_collate_fn(batch):
    images = []
    targets = []
    fnames = []
    
    for image, target, fname in batch:
        images.append(image)
        targets.append(target)
        fnames.append(fname)
        
    return images, targets, fnames

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

                
class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_map = None
        self.early_stop = False
        self.best_epoch = 0
        self.require_save = False

    def __call__(self, map_score, epoch):
        if self.best_map is None:
            self.best_map = map_score
            self.best_epoch = epoch
        elif map_score < self.best_map + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            self.require_save = False
        else:
            self.best_map = map_score
            self.best_epoch = epoch
            self.counter = 0
            self.require_save = True

def save_final_state(best_weights_path,writer, train_info_path, best_model_path, train_config, 
                    early_stopping, training_start_time, timestamp, log_dir):
    """
    Save final training state, logs, and best model
    """
    # Close tensorboard writer
    writer.close()
    
    # Calculate total training time
    total_time = time.time() - training_start_time
    
    if early_stopping.best_map is not None and early_stopping.best_epoch is not None:
        # Save training summary
        with open(train_info_path, 'a') as f:
            f.write(f"\nTotal training time: {total_time:.2f}s\n")
            f.write(f"Best mAP: {early_stopping.best_map:.4f} at epoch {early_stopping.best_epoch}\n")
            
        # Copy the best model weights only (not full checkpoint)
        """
        if best_model_path and os.path.exists(best_model_path):
            try:
                # Load the checkpoint
                checkpoint = torch.load(best_model_path)
                model_state_dict = checkpoint['model_state_dict']
                
                # Save just the model state dict
                best_model_final_path = os.path.join(
                    train_config['task_name'],
                    train_config['ckpt_name']
                )
                os.makedirs(os.path.dirname(best_model_final_path), exist_ok=True)
                torch.save(model_state_dict, best_model_final_path)
                print(f"Best model weights saved to: {best_model_final_path}")
                
            except Exception as e:
                print(f"Error saving best model weights: {str(e)}")
                    
        else:
            with open(train_info_path, 'a') as f:
                f.write(f"\nTotal training time: {total_time:.2f}s\n")
            f.write("No best mAP recorded yet\n")
        """
        
        shutil.copy(best_weights_path, os.path.join(
                    train_config['task_name'],
                    train_config['ckpt_name']))
        print(os.path.join(
                    train_config['task_name'],
                    train_config['ckpt_name']))
    
    # Create zip file of logs
    logs_zip_path = os.path.join(train_config['task_name'], 
                                f'tensorboard_logs_{timestamp}.zip')
    zip_logs(log_dir, logs_zip_path)
    print(f"\nTensorBoard logs saved to: {logs_zip_path}")
    
    # Print final training status
    print(f"Best model checkpoint at: {best_model_path}")
    if early_stopping.best_map is not None and early_stopping.best_epoch is not None:
        print(f"Best mAP: {early_stopping.best_map:.4f} at epoch {early_stopping.best_epoch}")



def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################
   
   

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.forcecpu:
        device = torch.device('cpu')
    
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"run_{timestamp}"
    
    # Initialize TensorBoard writer and early stopping
    log_dir = os.path.join(train_config['task_name'], 'logs', run_name)
    task_dir = train_config['task_name']
    os.makedirs(task_dir, exist_ok=True)
    os.makedirs(os.path.join(task_dir, 'logs'), exist_ok=True)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
   
    config_save_path = os.path.join(log_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)
    
    writer = SummaryWriter(log_dir)
    early_stopping = EarlyStopping()  # Initialize early stopping

    
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    
    voc = VOCDataset('train',
                     im_dir=dataset_config['im_train_path'],
                     ann_dir=dataset_config['ann_train_path'],
                     depth_dir=dataset_config['depth_path'])
    
    val_dataset = VOCDataset('train',
                     im_dir=dataset_config['im_train_path'],
                     ann_dir=dataset_config['ann_train_path'],
                     depth_dir=dataset_config['depth_path'])
    
    val_dataloader = DataLoader(voc,
                               batch_size=1,
                               shuffle=True,
                               num_workers=2,
                               ) 

    train_dataset = DataLoader(voc,
                               batch_size=1,
                               shuffle=True,
                               num_workers=2,
                               ) 
    
    faster_rcnn_model = FasterRCNN(model_config,
                                   num_classes=dataset_config['num_classes'])
    faster_rcnn_model.train()
    faster_rcnn_model.to(device)

    if not os.path.exists(train_config['task_name']):
        os.mkdirs(train_config['task_name'])
    optimizer = torch.optim.SGD(lr=train_config['lr'],
                                params=filter(lambda p: p.requires_grad,
                                              faster_rcnn_model.parameters()),
                                weight_decay=5E-4,
                                momentum=0.9)
    scheduler = MultiStepLR(optimizer, milestones=train_config['lr_steps'], gamma=0.1)
    
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
    best_model_path = None
    best_weights_path = os.path.join(
                    train_config['task_name'],
                    train_config['ckpt_name'] + "best"
                )
    try:
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            rpn_classification_losses = []
            rpn_localization_losses = []
            frcnn_classification_losses = []
            frcnn_localization_losses = []
            optimizer.zero_grad()
            val_rpn_classification_losses = []
            val_rpn_localization_losses = []
            val_frcnn_classification_losses = []
            val_frcnn_localization_losses = []
            
            # Training loop
            for im, target, fname in tqdm(train_dataset):
                im = im.float().to(device)
                target['bboxes'] = target['bboxes'].float().to(device)
                target['labels'] = target['labels'].long().to(device)
                rpn_output, frcnn_output = faster_rcnn_model(im, target)
                
                rpn_loss = rpn_output['rpn_classification_loss'] + rpn_output['rpn_localization_loss']
                frcnn_loss = frcnn_output['frcnn_classification_loss'] + frcnn_output['frcnn_localization_loss']
                loss = rpn_loss + frcnn_loss

                # Move losses to CPU before converting to numpy
                rpn_classification_losses.append(rpn_output['rpn_classification_loss'].detach().cpu().item())
                rpn_localization_losses.append(rpn_output['rpn_localization_loss'].detach().cpu().item())
                frcnn_classification_losses.append(frcnn_output['frcnn_classification_loss'].detach().cpu().item())
                frcnn_localization_losses.append(frcnn_output['frcnn_localization_loss'].detach().cpu().item())
                
                loss = loss / acc_steps
                loss.backward()
                if step_count % acc_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                step_count += 1
                global_step += 1

            # Validation loop
            with torch.no_grad():
                for val_im, val_target, val_fname in tqdm(val_dataloader):
                    val_im = val_im.float().to(device)
                    val_target['bboxes'] = val_target['bboxes'].float().to(device)
                    val_target['labels'] = val_target['labels'].long().to(device)
                    
                    val_rpn_output, val_frcnn_output = faster_rcnn_model(val_im, val_target)
                    
                    # Move validation losses to CPU before appending
                    val_rpn_classification_losses.append(val_rpn_output['rpn_classification_loss'].cpu().item())
                    val_rpn_localization_losses.append(val_rpn_output['rpn_localization_loss'].cpu().item())
                    val_frcnn_classification_losses.append(val_frcnn_output['frcnn_classification_loss'].cpu().item())
                    val_frcnn_localization_losses.append(val_frcnn_output['frcnn_localization_loss'].cpu().item())

            # Calculate metrics (now safe to use numpy as all values are on CPU)
            epoch_rpn_cls_loss = np.mean(rpn_classification_losses)
            epoch_rpn_loc_loss = np.mean(rpn_localization_losses)
            epoch_frcnn_cls_loss = np.mean(frcnn_classification_losses)
            epoch_frcnn_loc_loss = np.mean(frcnn_localization_losses)
            
            val_epoch_rpn_cls_loss = np.mean(val_rpn_classification_losses)
            val_epoch_rpn_loc_loss = np.mean(val_rpn_localization_losses)
            val_epoch_frcnn_cls_loss = np.mean(val_frcnn_classification_losses)
            val_epoch_frcnn_loc_loss = np.mean(val_frcnn_localization_losses)

            # Log metrics to TensorBoard
            writer.add_scalar('Loss/Epoch/RPN_Classification', epoch_rpn_cls_loss, epoch)
            writer.add_scalar('Loss/Epoch/RPN_Localization', epoch_rpn_loc_loss, epoch)
            writer.add_scalar('Loss/Epoch/FRCNN_Classification', epoch_frcnn_cls_loss, epoch)
            writer.add_scalar('Loss/Epoch/FRCNN_Localization', epoch_frcnn_loc_loss, epoch)
            writer.add_scalar('Loss/Epoch/RPN_Classification_validation', val_epoch_rpn_cls_loss, epoch)
            writer.add_scalar('Loss/Epoch/RPN_Localization_validation', val_epoch_rpn_loc_loss, epoch)
            writer.add_scalar('Loss/Epoch/FRCNN_Classification_validation', val_epoch_frcnn_cls_loss, epoch)
            writer.add_scalar('Loss/Epoch/FRCNN_Localization_validation', val_epoch_frcnn_loc_loss, epoch)
            writer.add_scalar('Loss/Epoch/total_localization_validation', val_epoch_rpn_loc_loss + val_epoch_frcnn_loc_loss, epoch)

            

         

            with torch.no_grad():  # Disable gradient computation
                for val_im, val_target, val_fname in tqdm(val_dataloader):
                    val_im = val_im.float().to(device)
                    val_target['bboxes'] = val_target['bboxes'].float().to(device)
                    val_target['labels'] = val_target['labels'].long().to(device)
                    
                    # Forward pass
                    val_rpn_output, val_frcnn_output = faster_rcnn_model(val_im, val_target)
                    val_rpn_classification_losses.append(val_rpn_output['rpn_classification_loss'])
                    val_rpn_localization_losses.append(val_rpn_output['rpn_localization_loss'])
                    val_frcnn_classification_losses.append(val_frcnn_output['frcnn_classification_loss'])
                    val_frcnn_localization_losses.append(val_frcnn_output['frcnn_localization_loss'])

            val_loss = 0.0
            val_loss /= len(val_dataset)
            val_epoch_rpn_cls_loss = np.mean(val_rpn_classification_losses)
            val_epoch_rpn_loc_loss = np.mean(val_rpn_localization_losses)
            val_epoch_frcnn_cls_loss = np.mean(val_frcnn_classification_losses)
            val_epoch_frcnn_loc_loss = np.mean(val_frcnn_localization_losses)

            # Log validation loss to TensorBoard
            writer.add_scalar('Loss/Epoch/RPN_Classification_validation', val_epoch_rpn_cls_loss, epoch)
            writer.add_scalar('Loss/Epoch/RPN_Localization_validation', val_epoch_rpn_loc_loss, epoch)
            writer.add_scalar('Loss/Epoch/FRCNN_Classification_validation',  val_epoch_frcnn_cls_loss, epoch)
            writer.add_scalar('Loss/Epoch/FRCNN_Localization_validation',  val_epoch_frcnn_loc_loss, epoch)
            writer.add_scalar('Loss/Epoch/total_localization_validation',  val_epoch_rpn_loc_loss +  val_epoch_frcnn_loc_loss, epoch )
                
            print(f'train loss  {epoch_frcnn_loc_loss}')
            print(f'val loss  {val_epoch_frcnn_loc_loss}')
            
            # Evaluate mAP and handle early stopping
            #save because evaluate map uses the weights 
            checkpoint_base = os.path.abspath(task_dir)
            checkpoint_path = os.path.join(checkpoint_base, train_config['ckpt_name'])
            
            # When saving the model
            #optimizer.step()
            #optimizer.zero_grad()
            torch.save(faster_rcnn_model.state_dict(), os.path.join(train_config['task_name'],
                                                                train_config['ckpt_name']))
            loss_output = ''
            loss_output += 'RPN Classification Loss : {:.4f}'.format(np.mean(rpn_classification_losses))
            loss_output += ' | RPN Localization Loss : {:.4f}'.format(np.mean(rpn_localization_losses))
            loss_output += ' | FRCNN Classification Loss : {:.4f}'.format(np.mean(frcnn_classification_losses))
            loss_output += ' | FRCNN Localization Loss : {:.4f}'.format(np.mean(frcnn_localization_losses))
            print(loss_output)
            #torch.save(faster_rcnn_model.state_dict(), checkpoint_path)
            # torch.save(faster_rcnn_model.state_dict(), os.path.join(train_config['task_name'],
            #                                                   train_config['ckpt_name']))

            map_score = evaluate_map(args,validation_set=True)
            writer.add_scalar('map', map_score, epoch)
            early_stopping(map_score, epoch)
            
            if early_stopping.require_save:
                 torch.save(faster_rcnn_model.state_dict(), best_weights_path)
            faster_rcnn_model.train()
            
            # Save checkpoint
            checkpoint_path = os.path.join(log_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': faster_rcnn_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'global_step': global_step,
                'map_score': map_score,
                'config': config
            }, checkpoint_path)
            
            # Update best model path if this is the best mAP
            if early_stopping.best_epoch == epoch:
                if best_model_path and os.path.exists(best_model_path):
                    os.remove(best_model_path)
                best_model_path = checkpoint_path
            
            # Update training info file
            with open(train_info_path, 'a') as f:
                f.write(f"\nEpoch {epoch} completed in {epoch_time:.2f}s\n")
                f.write(f"Average losses:\n")
                f.write(f"  RPN Classification: {epoch_rpn_cls_loss:.4f}\n")
                f.write(f"  RPN Localization: {epoch_rpn_loc_loss:.4f}\n")
                f.write(f"  FRCNN Classification: {epoch_frcnn_cls_loss:.4f}\n")
                f.write(f"  FRCNN Localization: {epoch_frcnn_loc_loss:.4f}\n")
                f.write(f"  mAP: {map_score:.4f}\n")
            
            scheduler.step()
            
            # Check for early stopping
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch}. Best mAP: {early_stopping.best_map:.4f} at epoch {early_stopping.best_epoch}")
                break
        
    except Exception as e:
        print(f"Training interrupted: {str(e)}")
    finally:
         save_final_state( best_weights_path ,
        writer=writer,
        train_info_path=train_info_path,
        best_model_path=best_model_path,
        train_config=train_config,
        early_stopping=early_stopping,
        training_start_time=training_start_time,
        timestamp=timestamp,
        log_dir=log_dir,
    
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for faster rcnn training')
    parser.add_argument('--config', dest='config_path',
                        default='config/conf.yaml', type=str)
    parser.add_argument('--forcecpu', action='store_true',
                        help='Force using CPU even if CUDA is available')
    args = parser.parse_args()
    train(args)
