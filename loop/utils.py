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

from data.dataloader import COCOLoader, collate_function



# main loop ============================================================================
def data_loader_manager(config_dict, seed, transforms):
    
    # producing generator with seed for data loader repeatability
    gen = torch.Generator()
    gen.manual_seed(seed)

    # get required datasets
    if config_dict['TRAIN']:
        # training dataset and loader
        train_data = COCOLoader(
                        root = config_dict['train_dir'], 
                        json_root = config_dict['train_json'], # put back comma when augmenting 
                        transforms = transforms
                        )
        train_loader = torch.utils.data.DataLoader(
                        train_data,
                        batch_size = config_dict['batch_size'],
                        shuffle = config_dict['loader_shuffle'],
                        num_workers = config_dict['loader_workers'],
                        worker_init_fn = seed_worker,
                        generator = gen,
                        collate_fn = collate_function)

        # validate dataset and loader
        validate_data = COCOLoader(
                        root = config_dict['validate_dir'], 
                        json_root = config_dict['validate_json'],
                        ) # no transforms in validation
        validate_loader = torch.utils.data.DataLoader(
                        validate_data,
                        batch_size = config_dict['batch_size'],
                        shuffle = config_dict['loader_shuffle'],
                        num_workers = config_dict['loader_workers'],
                        worker_init_fn = seed_worker,
                        generator = gen,
                        collate_fn = collate_function)

    if config_dict['TEST']:
        test_data = COCOLoader(
                        root = config_dict['test_dir'], 
                        json_root = config_dict['test_json'],
                        ) # no transforms in test
        test_loader = torch.utils.data.DataLoader(
                        test_data,
                        batch_size = config_dict['batch_size'],
                        shuffle = config_dict['loader_shuffle'],
                        num_workers = config_dict['loader_workers'],
                        worker_init_fn = seed_worker,
                        generator = gen,
                        collate_fn = collate_function)

    # retruning loaders
    if config_dict['TRAIN'] and config_dict['TEST']:
        return train_loader, validate_loader, test_loader
    elif config_dict['TRAIN']:
        return train_loader, validate_loader, None
    else:
        return None, None, test_loader

def model_saver(epoch, model, optimizer, best_result, mAP, path):
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

    if mAP < best_result:   # swapped for loss value: change this back
        best_result = mAP
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
    #print("Worker ID:", info.id, "Worker Seed:",worker_seed)

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