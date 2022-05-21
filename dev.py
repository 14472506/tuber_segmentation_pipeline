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
from dataloader import COCOLoader, collate_function
from models import model_selector
from optimizer import optimizer_selector, lr_scheduler_selector
from engine import train_one_epoch, validate_one_epoch, fps_evaluate, segment_instance
from transforms import transform_selector
from utils import model_saver, make_dir, time_converter
from coco_evaluation import evaluate
import configs
from plotter import plot_lr_loss, plot_precision_recall
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
                num_workers = 4,
                collate_fn = collate_function)

break_key = False
count = 0

points = []

for images, targets in train_loader:

    if break_key:
        break

    #print("tuple output")
    #print(images)

    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    masks_array = targets[0]["masks"].cpu().numpy()
    for i in masks_array:

        ## making white mask 
        #white = [255, 255, 255]
        #r = np.zeros_like(i).astype(np.uint8)
        #g = np.zeros_like(i).astype(np.uint8)
        #b = np.zeros_like(i).astype(np.uint8)
        #r[i == 1], g[i == 1], b[i == 1] = white
        #white_mask = np.stack([r, g, b], axis=2)

        # keep this for future work
        #mask_loc = np.where(i == 1)
        ##mask_points = np.concatenate(mask_loc[0], mask_loc[1])
        #mask_points = np.transpose(np.vstack(mask_loc))

        cnt, _ = cv2.findContours(i, 1, 2)
        cnt = cnt[0]

        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        point = [cx, cy]
        points.append(point)
        
        #cv2.imshow("mask", coloured_mask)
        #cv2.waitKey(0)

        break

    if count >= 1:
        break_key = True
    
    count +=1

error = centroid_error(points)

print(error)
print(points)

