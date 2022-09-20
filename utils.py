"""
Detials
"""
# =============================================================================================== #
# Imports
# =============================================================================================== #
import os
import errno
import torch
import torch.distributed as dist
import random
import numpy as np

from dataloader import COCOLoader, collate_function

# =============================================================================================== #
# Functions
# =============================================================================================== #
def make_dir(path):
    """
    Detials
    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def model_saver(epoch, model, optimizer, path, pth_name):
    """
    Detials
    """
    # check if output dir exists, if not make dir
    make_dir(path)

    # making save dictionary for model.pth
    checkpoint = {
        "epoch": epoch + 1,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    # saving last model
    last_model_path = path + "/" + pth_name
    torch.save(checkpoint, last_model_path)


def all_gather(data):
    """
    Detials
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list

def get_world_size():
    """
    Detials
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def is_dist_avail_and_initialized():
    """
    Detials
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def time_converter(seconds):
    """
    Detials
    """
    hours = seconds // 3600
    seconds %= 3600
    min = seconds // 60
    seconds %= 60
    
    return("%02d:%02d:%02d" %(hours, min, seconds))