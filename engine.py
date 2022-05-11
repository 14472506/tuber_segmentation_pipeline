"""
Details
"""
# ===========================
# Import libraties/packages
# ===========================
from email.mime import image
import torch
import numpy as np
import math
import sys
import json

# ===========================
# functions
# ===========================
# train 
def train_one_epoch(train_loader, model, device, optimizer, print_freq, iter_count):
    """
    train details
    """
    # set/ensure model output is configured for trainingtrain_loader, model, device, optimizer,
    model.train()

    # loss_collection
    loss_col = []

    # looping through dataset
    for images, targets in train_loader:

        # currently loading images and targets to device. REPLACE this with colate function and
        # and more effient method of sending images and target to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # passing batch to model to carry out forward
        # loss_dict contains loss tensors model losses. meta data such as loss functions and 
        # grad function also returned from model
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # collect losses
        loss_col.append(losses.item())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # printing results 
        if iter_count % print_freq == 0:
            print("[iter: %s] total_loss: %s" %(iter_count, losses.item()))
        iter_count += 1

    # return losses
    return loss_col, iter_count

def validate_one_epoch(validation_loader, model, device):
    """
    validation details
    """
    # this is questionable and should be checked.
    model.train()

    # validation collection
    val_col = []

    # disables gradient calculation
    with torch.no_grad():
        for images, targets in validation_loader:

            # currently loading images and targets to device. REPLACE this with colate function and
            # and more effient method of sending images and target to device
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # getting losses
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # append losses to val_col
            val_col.append(losses.item())
        
    return(val_col)





def evaluate():
    print('dave')