# ============================
# Importing packages/libraries 
# ============================
# pytorch imports

from email.mime import image
import json
from logging import root

from engine import train_one_epoch
from optimizer import optimizer_selector
import torch 
import torchvision.transforms as T

# suporting library imports
import numpy as np
import random
import time

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
                batch_size = 2,
                shuffle = True,
                num_workers = 4,
                collate_fn = collate_function)

break_key = False
for images, targets in train_loader:

    if break_key:
        break

    #print("tuple output")
    #print(images)

    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    #print("tensor output")
    #print(images)

    break_key = True