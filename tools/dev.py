# ============================
# Importing packages/libraries 
# ============================
# pytorch imports

from math import sqrt
from email.mime import image
import json
from logging import root

from pkg_resources import ensure_directory

from engine import train_one_epoch
from optimizer import optimizer_selector
import torch 
import torchvision.transforms as T

# suporting library imports
import numpy as np
import random
import time
import cv2

# package imports
from data.dataloader import COCOLoader, collate_function
from models.models import model_selector
from optimizer import optimizer_selector, lr_scheduler_selector
from engine import train_one_epoch, validate_one_epoch, fps_evaluate, segment_instance
from data.transforms import transform_selector
from utils import model_saver, make_dir, time_converter
from evaluation.coco_evaluation import evaluate
import config.configs as configs
from evaluation.plotter import plot_lr_loss, plot_precision_recall
# ============================
# Train_net implementation
# ============================
def centroid_error(points):
    
    x1 = points[0][0]
    x2 = points[1][0]
    y1 = points[0][1]
    y2 = points[1][1]
    
    error = abs(sqrt(abs((x1 - x2)**2 + abs(y1 - y2)**2))) 

    return error

#################################

# configuring device for cpu or gpu if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# fixing the random seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
    

transforms = None
    

# training dataset and loader
train_data = COCOLoader(
                root = "data/jersey_royal_dataset/test", 
                json_root = "data/jersey_royal_dataset/test/test.json", # put back comma when augmenting 
                transforms = transforms
                )
train_loader = torch.utils.data.DataLoader(
                train_data,
                batch_size = 1,
                shuffle = True,
                num_workers = 0,
                collate_fn = collate_function)

break_key = False
count = 0

points = []

for images, targets in train_loader:

    if break_key:
        break
        
    targ_masks = targets[0]['masks'].detach().cpu().numpy()
    
    for mask in targ_masks:
        points = np.where(mask == 1)
        x_points = points[1]
        y_points = points[0]
        
        x_points.sort()
        y_points.sort()
        
        x_idx = int((len(x_points) - 1)/2)
        y_idx = int((len(y_points) - 1)/2)
        
        centroid = [x_points[x_idx], y_points[y_idx]]
        print(centroid)
        
        break

    break

    
    #images = list(image.to(device) for image in images)
    #targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    


