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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

def collate_fn(batch):
    """
    Collate function that handles variable numbers of bounding boxes per image.
    
    Args:
        batch (list): List of tuples (image_tensor, annotations_dict, image_path)
    
    Returns:
        tuple: (batch of processed images, dict containing batched bboxes and labels)
    """
    images = []
    all_boxes = []
    all_labels = []
  
   
    # First pass to get max number of boxes
    max_boxes = 0
    for image, annotations in batch:
        num_boxes = len(annotations['bboxes'])
        max_boxes = max(max_boxes, num_boxes)
    
    # Second pass to pad boxes and labels
    for image, annotations in batch:
        images.append(image)
       
        # Get current boxes and labels
        boxes = annotations['bboxes']
        labels = annotations['labels']
        num_boxes = len(boxes)
        
        # Pad boxes if necessary
        if num_boxes < max_boxes:
            # Pad boxes with zeros
            padding = torch.zeros((max_boxes - num_boxes, 4), dtype=boxes.dtype, device=boxes.device)
            boxes = torch.cat([boxes, padding], dim=0)
            
            # Pad labels with zeros or background class (usually 0)
            label_padding = torch.zeros(max_boxes - num_boxes, dtype=labels.dtype, device=labels.device)
            labels = torch.cat([labels, label_padding], dim=0)
        
        all_boxes.append(boxes)
        all_labels.append(labels)
        
    
    # Stack everything
    images = torch.stack(images, dim=0)
    boxes = torch.stack(all_boxes, dim=0)
    labels = torch.stack(all_labels, dim=0)
   
    
    # Process images and boxes
    #images, boxes = normalize_resize_image_and_boxes(images, boxes)
    print("yo2")
    print(images.shape)
    return images, {'bboxes': boxes, 'labels': labels}

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

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################
    if args.forcecpu:
        device = torch.device('cpu')
    
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']
    
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
                               batch_size=6,
                               shuffle=True,
                               num_workers=2,
                               collate_fn=collate_fn
                           
                               )
    
    faster_rcnn_model = FasterRCNN(model_config,
                                   num_classes=dataset_config['num_classes'])
    faster_rcnn_model.train()
    faster_rcnn_model.to(device)

    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
    optimizer = torch.optim.SGD(lr=train_config['lr'],
                                params=filter(lambda p: p.requires_grad,
                                              faster_rcnn_model.parameters()),
                                weight_decay=5E-4,
                                momentum=0.9)
    scheduler = MultiStepLR(optimizer, milestones=train_config['lr_steps'], gamma=0.1)
    
    acc_steps = train_config['acc_steps']
    num_epochs = train_config['num_epochs']
    step_count = 1

    for i in range(num_epochs):
        rpn_classification_losses = []
        rpn_localization_losses = []
        frcnn_classification_losses = []
        frcnn_localization_losses = []
        optimizer.zero_grad()
        
        for im, target in tqdm(train_dataset):
            im = im.float().to(device)
            target['bboxes'] = target['bboxes'].float().to(device)
            target['labels'] = target['labels'].long().to(device)
            rpn_output, frcnn_output = faster_rcnn_model(im, target)
            
            rpn_loss = rpn_output['rpn_classification_loss'] + rpn_output['rpn_localization_loss']
            frcnn_loss = frcnn_output['frcnn_classification_loss'] + frcnn_output['frcnn_localization_loss']
            loss = rpn_loss + frcnn_loss
    
            rpn_classification_losses.append(rpn_output['rpn_classification_loss'].item())
            rpn_localization_losses.append(rpn_output['rpn_localization_loss'].item())
            frcnn_classification_losses.append(frcnn_output['frcnn_classification_loss'].item())
            frcnn_localization_losses.append(frcnn_output['frcnn_localization_loss'].item())
            loss = loss / acc_steps
            loss.backward()
            if step_count % acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            step_count += 1
        print('Finished epoch {}'.format(i))
        torch.save(faster_rcnn_model.state_dict(), os.path.join(train_config['task_name'],
                                                                train_config['ckpt_name']))
        loss_output = ''
        loss_output += 'RPN Classification Loss : {:.4f}'.format(np.mean(rpn_classification_losses))
        loss_output += ' | RPN Localization Loss : {:.4f}'.format(np.mean(rpn_localization_losses))
        loss_output += ' | FRCNN Classification Loss : {:.4f}'.format(np.mean(frcnn_classification_losses))
        loss_output += ' | FRCNN Localization Loss : {:.4f}'.format(np.mean(frcnn_localization_losses))
        print(loss_output)
        scheduler.step()
    print('Done Training...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for faster rcnn training')
    parser.add_argument('--config', dest='config_path',
                        default='config/conf.yaml', type=str)
    parser.add_argument('--forcecpu', action='store_true',
                        help='Force using CPU even if CUDA is available')
    args = parser.parse_args()
    train(args)
