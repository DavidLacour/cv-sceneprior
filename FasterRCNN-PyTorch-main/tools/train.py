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


def load_existing_weights(model, weights_path, device, logger=None):
    """
    Check for and load existing model weights if they exist.
    
    Args:
        model: The model to load weights into
        weights_path: Path to the weights file
        device: torch device (cuda/cpu)
        logger: Optional logging function or file
    
    Returns:
        bool: True if weights were loaded successfully, False otherwise
    """
    try:
        if os.path.exists(weights_path):
            print(f"Found existing weights at {weights_path}")
            if logger:
                logger.write(f"Found existing weights at {weights_path}\n")
            
            # Load the state dict
            state_dict = torch.load(weights_path, map_location=device)
            
            # Check if state_dict is wrapped in a checkpoint
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            
            # Load the weights
            model.load_state_dict(state_dict)
            print("Successfully loaded existing model weights")
            if logger:
                logger.write("Successfully loaded existing model weights\n")
            return True
            
        else:
            print(" ################### No existing weights found. Starting from scratch.")
            if logger:
                logger.write("No existing weights found. Starting from scratch.\n")
            return False
    except Exception as e:
        print(f"Error loading weights: {str(e)}")
        print("Starting from scratch due to loading error.")
        if logger:
            logger.write(f"Error loading weights: {str(e)}\n")
            logger.write("Starting from scratch due to loading error.\n")
        return False


def cleanup_old_checkpoints(log_dir, current_epoch, patience):
    """
    Remove checkpoint files that are older than (current_epoch - patience)
    This ensures we keep a rolling window of recent checkpoints while cleaning up older ones
    
    Args:
        log_dir: Directory containing checkpoint files
        current_epoch: Current training epoch
        patience: Early stopping patience value
    """
    oldest_epoch_to_keep = max(0, current_epoch - patience)
    
    for filename in os.listdir(log_dir):
        if filename.startswith("checkpoint_epoch_") and filename.endswith(".pth"):
            try:
                epoch_num = int(filename.split("_")[-1].split(".")[0])
                checkpoint_path = os.path.join(log_dir, filename)
                
                # Remove checkpoints older than our window
                if epoch_num < oldest_epoch_to_keep:
                    try:
                        os.remove(checkpoint_path)
                        print(f"Removed old checkpoint: {filename}")
                    except Exception as e:
                        print(f"Error removing checkpoint {filename}: {str(e)}")
            except ValueError:
                # Skip files that don't match our naming pattern
                continue



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
    def __init__(self, patience=60, min_delta=0.0001):
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
    """
    # Create zip file of logs
    logs_zip_path = os.path.join(train_config['task_name'], 
                                f'tensorboard_logs_{timestamp}.zip')
    zip_logs(log_dir, logs_zip_path)
    print(f"\nTensorBoard logs saved to: {logs_zip_path}")
    """
    
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
    patience = train_config['patience']

    
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
    
    train_dataset = VOCDataset('train',
                     im_dir=dataset_config['im_train_path'],
                     ann_dir=dataset_config['ann_train_path'],
                     depth_dir=dataset_config['depth_path'])
    
    val_dataset = VOCDataset('val',
                     im_dir=dataset_config['im_val_path'],
                     ann_dir=dataset_config['ann_val_path'],
                     depth_dir=dataset_config['depth_path'])
    
    val_dataloader = DataLoader(val_dataset,
                               batch_size=1,
                               shuffle=True,
                               num_workers=2,
                               ) 

    train_dataloader = DataLoader(train_dataset,
                               batch_size=1,
                               shuffle=True,
                               num_workers=2,
                               ) 
    
    test_dataset = VOCDataset('test',
                     im_dir=dataset_config['im_test_path'],
                     ann_dir=dataset_config['ann_test_path'],
                     depth_dir=dataset_config['depth_path'])
    
    test_dataloader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=True,
                             num_workers=2)

    
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
        f.write(f"Dataset size: {len(train_dataset)}\n")
    

    acc_steps = train_config['acc_steps']
    num_epochs = train_config['num_epochs']
    step_count = 1
    global_step = 0
    training_start_time = time.time()
    best_model_path = None
    best_weights_path = os.path.join(
                    train_config['task_name'],
                     "best" + train_config['ckpt_name'] 
                )
    try:
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            train_rpn_classification_losses = []
            train_rpn_localization_losses = []
            train_frcnn_classification_losses = []
            train_frcnn_localization_losses = []
            val_rpn_classification_losses = []
            val_rpn_localization_losses = []
            val_frcnn_classification_losses = []
            val_frcnn_localization_losses = []
            test_rpn_classification_losses = []
            test_rpn_localization_losses = []
            test_frcnn_classification_losses = []
            test_frcnn_localization_losses = []


            optimizer.zero_grad()
            
            for im, target, fname in tqdm(train_dataloader):
                im = im.float().to(device)
                target['bboxes'] = target['bboxes'].float().to(device)
                target['labels'] = target['labels'].long().to(device)
                rpn_output, frcnn_output = faster_rcnn_model(im, target)
                
                rpn_loss = rpn_output['rpn_classification_loss'] + rpn_output['rpn_localization_loss']
                frcnn_loss = frcnn_output['frcnn_classification_loss'] + frcnn_output['frcnn_localization_loss']
                loss = rpn_loss + frcnn_loss
                
                # Accumulate losses for epoch-level logging
                train_rpn_classification_losses.append(rpn_output['rpn_classification_loss'].item())
                train_rpn_localization_losses.append(rpn_output['rpn_localization_loss'].item())
                train_frcnn_classification_losses.append(frcnn_output['frcnn_classification_loss'].item())
                train_frcnn_localization_losses.append(frcnn_output['frcnn_localization_loss'].item())

                loss = loss / acc_steps
                loss.backward()
                if step_count % acc_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                step_count += 1
                global_step += 1
            
            # faster_rcnn_model.eval()  # Set model to evaluation mode crahes
            val_loss = 0.0
            with torch.no_grad():  # Disable gradient computation
                for val_im, val_target, fname in tqdm(val_dataloader):
                    val_im = val_im.float().to(device)
                    val_target['bboxes'] = val_target['bboxes'].float().to(device)
                    val_target['labels'] = val_target['labels'].long().to(device)
                    
                    # Forward pass
                    val_rpn_output, val_frcnn_output = faster_rcnn_model(val_im, val_target)
                    val_rpn_classification_losses.append(val_rpn_output['rpn_classification_loss'])
                    val_rpn_localization_losses.append(val_rpn_output['rpn_localization_loss'])
                    val_frcnn_classification_losses.append(val_frcnn_output['frcnn_classification_loss'])
                    val_frcnn_localization_losses.append(val_frcnn_output['frcnn_localization_loss'])

                for test_im, test_target, fname in tqdm(test_dataloader):
                    test_im = test_im.float().to(device)
                    test_target['bboxes'] = test_target['bboxes'].float().to(device)
                    test_target['labels'] = test_target['labels'].long().to(device)
                    
                    # Forward pass
                    test_rpn_output, test_frcnn_output = faster_rcnn_model(test_im, test_target)
                    test_rpn_classification_losses.append(test_rpn_output['rpn_classification_loss'])
                    test_rpn_localization_losses.append(test_rpn_output['rpn_localization_loss'])
                    test_frcnn_classification_losses.append(test_frcnn_output['frcnn_classification_loss'])
                    test_frcnn_localization_losses.append(test_frcnn_output['frcnn_localization_loss'])


                    
            
            # Average validation loss
            val_loss /= len(val_dataset)
            faster_rcnn_model.train()
           
            
           

            epoch_time = time.time() - epoch_start_time
            print(f'Finished epoch {epoch}, time taken: {epoch_time:.2f}s')
            
            # Find this section in train.py and modify the validation loss calculations:

            # Calculate and log epoch metrics
            train_epoch_rpn_cls_loss = np.mean(train_rpn_classification_losses)
            train_epoch_rpn_loc_loss = np.mean(train_rpn_localization_losses)
            train_epoch_frcnn_cls_loss = np.mean(train_frcnn_classification_losses)
            train_epoch_frcnn_loc_loss = np.mean(train_frcnn_localization_losses)
            with torch.no_grad():
                train_total_rpn_loss =  train_epoch_rpn_cls_loss  + train_epoch_rpn_loc_loss
                train_total_frcnn_loss = train_epoch_frcnn_cls_loss + train_epoch_frcnn_loc_loss
                train_total_overall_loss = train_total_rpn_loss + train_total_frcnn_loss
            
            writer.add_scalar('train_Loss/Epoch/RPN_Classification', train_epoch_rpn_cls_loss, epoch)
            writer.add_scalar('train_Loss/Epoch/RPN_Localization', train_epoch_rpn_loc_loss, epoch)
            writer.add_scalar('train_Loss/Epoch/FRCNN_Classification', train_epoch_frcnn_cls_loss, epoch)
            writer.add_scalar('train_Loss/Epoch/FRCNN_Localization', train_epoch_frcnn_loc_loss, epoch)
            writer.add_scalar('train_Loss/Epoch/total_localization', train_epoch_rpn_loc_loss + train_epoch_frcnn_loc_loss, epoch)
            writer.add_scalar('train_Training/Epoch_Time', epoch_time, epoch)
            writer.add_scalar('train_Total_FRCNN_loss',train_total_frcnn_loss,epoch)
            writer.add_scalar('train_Total_RPN_loss',train_total_rpn_loss,epoch)
            writer.add_scalar('train_Total_overall_loss',train_total_overall_loss,epoch)




            # Move validation tensors to CPU before converting to numpy
            val_rpn_cls_losses = [loss.cpu().detach().numpy() for loss in val_rpn_classification_losses]
            val_rpn_loc_losses = [loss.cpu().detach().numpy() for loss in val_rpn_localization_losses]
            val_frcnn_cls_losses = [loss.cpu().detach().numpy() for loss in val_frcnn_classification_losses]
            val_frcnn_loc_losses = [loss.cpu().detach().numpy() for loss in val_frcnn_localization_losses]

            val_epoch_rpn_cls_loss = np.mean(val_rpn_cls_losses)
            val_epoch_rpn_loc_loss = np.mean(val_rpn_loc_losses)
            val_epoch_frcnn_cls_loss = np.mean(val_frcnn_cls_losses)
            val_epoch_frcnn_loc_loss = np.mean(val_frcnn_loc_losses)

            val_total_rpn_loss =  val_epoch_rpn_cls_loss  + val_epoch_rpn_loc_loss
            val_total_frcnn_loss = val_epoch_frcnn_cls_loss + val_epoch_frcnn_loc_loss
            val_total_overall_loss = val_total_rpn_loss + val_total_frcnn_loss

            # Log validation loss to TensorBoard
            writer.add_scalar('val_Loss/Epoch/RPN_Classification', val_epoch_rpn_cls_loss, epoch)
            writer.add_scalar('val_Loss/Epoch/RPN_Localization', val_epoch_rpn_loc_loss, epoch)
            writer.add_scalar('val_Loss/Epoch/FRCNN_Classification', val_epoch_frcnn_cls_loss, epoch)
            writer.add_scalar('val_Loss/Epoch/FRCNN_Localization', val_epoch_frcnn_loc_loss, epoch)
            writer.add_scalar('val_Loss/Epoch/total_localization', val_epoch_rpn_loc_loss + val_epoch_frcnn_loc_loss, epoch)
            writer.add_scalar('val_Total_FRCNN_loss',val_total_frcnn_loss,epoch)
            writer.add_scalar('val_Total_RPN_loss',val_total_rpn_loss,epoch)
            writer.add_scalar('val_Total_overall_loss',val_total_overall_loss,epoch)

            test_rpn_cls_losses = [loss.cpu().detach().numpy() for loss in test_rpn_classification_losses]
            test_rpn_loc_losses = [loss.cpu().detach().numpy() for loss in test_rpn_localization_losses]
            test_frcnn_cls_losses = [loss.cpu().detach().numpy() for loss in test_frcnn_classification_losses]
            test_frcnn_loc_losses = [loss.cpu().detach().numpy() for loss in test_frcnn_localization_losses]

            test_epoch_rpn_cls_loss = np.mean(test_rpn_cls_losses)
            test_epoch_rpn_loc_loss = np.mean(test_rpn_loc_losses)
            test_epoch_frcnn_cls_loss = np.mean(test_frcnn_cls_losses)
            test_epoch_frcnn_loc_loss = np.mean(test_frcnn_loc_losses)

            test_total_rpn_loss =  test_epoch_rpn_cls_loss  + test_epoch_rpn_loc_loss
            test_total_frcnn_loss = test_epoch_frcnn_cls_loss + test_epoch_frcnn_loc_loss
            test_total_overall_loss = test_total_rpn_loss + test_total_frcnn_loss

            # Log validation loss to TensorBoard
            writer.add_scalar('test_Loss/Epoch/RPN_Classification', test_epoch_rpn_cls_loss, epoch)
            writer.add_scalar('test_Loss/Epoch/RPN_Localization', test_epoch_rpn_loc_loss, epoch)
            writer.add_scalar('test_Loss/Epoch/FRCNN_Classification',test_epoch_frcnn_cls_loss, epoch)
            writer.add_scalar('test_Loss/Epoch/FRCNN_Localization', test_epoch_frcnn_loc_loss, epoch)
            writer.add_scalar('test_Loss/Epoch/total_localization', test_epoch_rpn_loc_loss + test_epoch_frcnn_loc_loss, epoch)
            writer.add_scalar('test_Total_FRCNN_loss',test_total_frcnn_loss,epoch)
            writer.add_scalar('test_Total_RPN_loss',test_total_rpn_loss,epoch)
            writer.add_scalar('test_Total_overall_loss',test_total_overall_loss,epoch)



            print(f"Epoch {epoch}: train frcnn Loss = {train_total_frcnn_loss}")
            print(f"Epoch {epoch}: Validation frcnn Loss = {val_total_frcnn_loss}")
            print(f"Epoch {epoch}: test frcnn Loss = {test_total_frcnn_loss}")

            # Evaluate mAP and handle early stopping
            # save because evaluae_map use saved weights 
            # if epoch > 9 : 
            model_path = os.path.join(train_config['task_name'], train_config['ckpt_name'])
            torch.save(faster_rcnn_model.state_dict(), model_path)

            #calulate map for each set 
            train_map_score = evaluate_map(args,validation_set=False,training_set=True)
            val_map_score = evaluate_map(args,validation_set=True,training_set=False)
            test_map_score = evaluate_map(args,validation_set=False,training_set=False)
           
          
            writer.add_scalar('train_map', train_map_score, epoch)
            writer.add_scalar('val_map', val_map_score, epoch)
            writer.add_scalar('test_map', test_map_score, epoch)


            print(f"Epoch {epoch}: train map = {train_map_score}")
            print(f"Epoch {epoch}: Val map = {val_map_score}")
            print(f"Epoch {epoch}: test map = {test_map_score}")

            early_stopping(val_map_score, epoch)
            faster_rcnn_model.train()
            # Save checkpoint
            checkpoint_path = os.path.join(log_dir, f"checkpoint_epoch_{epoch}.pth")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': faster_rcnn_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'global_step': global_step,
                'val_map_score': val_map_score,
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
                f.write(f"  RPN Classification: {train_epoch_rpn_cls_loss:.4f}\n")
                f.write(f"  RPN Localization: {train_epoch_rpn_loc_loss:.4f}\n")
                f.write(f"  FRCNN Classification: {train_epoch_frcnn_cls_loss:.4f}\n")
                f.write(f"  FRCNN Localization: {train_epoch_frcnn_loc_loss:.4f}\n")
                f.write(f"  val_mAP: {val_map_score:.4f}\n")
            
            scheduler.step()
            cleanup_old_checkpoints(log_dir, epoch, patience)
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