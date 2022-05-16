"""
Details
"""
# ===========================
# Import libraries/packages
# ===========================
import os
import torch
import errno
import torch.distributed as dist
import random
import numpy as np


# main loop ============================================================================
def model_saver(epoch, model, optimizer, best_result, val_loss, path):
    """
    Title       :

    Function    :

    Inputs      :

    Outputs     :

    Deps        :

    Edited by   :
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
    last_model_path = path + "/last_model.pth"
    torch.save(checkpoint, last_model_path)

    if val_loss < best_result:
        best_result = val_loss
        best_model_path = path + "/best_model.pth"
        torch.save(checkpoint, best_model_path)
    
    return(best_result)

def make_dir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def set_seed(seed):
    """
    Details
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def seed_worker(worker_id):
    """
    Details
    """
    info = torch.utils.data.get_worker_info()
    worker_seed =  torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed) 
    random.seed(worker_seed)
    print("Worker ID:", info.id, "Worker Seed:",worker_seed)

# coco eval =============================================================================
def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def time_converter(seconds):
    
    hours = seconds // 3600
    seconds %= 3600
    min = seconds // 60
    seconds %= 60
    
    return("%02d:%02d:%02d" %(hours, min, seconds))
