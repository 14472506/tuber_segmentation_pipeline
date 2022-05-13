"""
Title:      train_net.py

Fuction:    This script acts as the top level script for training all models for potato tuber
            segmentation. The script will configure the pipeline to the required parameters for
            the specific model being trained. If the script runs successful, the pipeline will
            produce a trained model

Edited by:  Bradley Hurst 
"""
# ============================
# Importing packages/libraries 
# ============================
# pytorch imports
from distutils.command.config import config
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
def main(config_dict, seed=42):
    """
    Title:      Main

    Function:   Main function carries out top level execution of all elements in the 
                training process

    Inputs:     - A dictionary of config parameters that is passed through the pipeline

    Outputs:    - The most recent model saved in a results location
                - Evaluation results from the model
                - A copy of the configuration dictionary used to produce the model

    Deps:       - COCOLoader in dataloader.py
                - collate_function in dataloader.py
                - model_selector in models.py
                - opimizer_selector in optimizer.py
                - lr_schedule_selector in optimizer.py
                - train_one_epoch in engine.py
                - val_one_epoch in engine.py
                - evaluate in engine.py
                - loop_saver from utils.py

    Edited by:  Bradley Hurst 
    """
    # configuring device for cpu or gpu if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # fixing the random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # configuring transforms for data loading
    if config_dict['transforms'] != "":
        transforms = transform_selector()
    else:
        transforms = None
    
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
                        collate_fn = collate_function)
    
    # get reqired model and set it to device
    model = model_selector(config_dict['model'], config_dict['num_classes'])
    model.to(device)
    #print(model)

    # get optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    #config_dict['model_params'] = params
    optimizer = optimizer_selector(config_dict['optimizer'],
                                   params,
                                   config_dict['optimizer_params'])

    # get learning rate scheduler
    if config_dict['lr_scheduler'] != "":
        lr_scheduler = lr_scheduler_selector(config_dict['lr_scheduler'])
    
    # define starting epoch
    start_epoch = 0

    # laoding model if required
    if config_dict['load'] != "":
        checkpoint = torch.load(config_dict["load"])
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]

    # save train params here: ys save the json here!
    config_save = config_dict['out_dir'] + "/model_configs.json"
    make_dir(config_dict['out_dir'])
    with open(config_save, 'w') as f:
        json.dump(config_dict, f)


    # training loop implementation
    if config_dict['TRAIN']: 
        # data trackers
        iter_count = 0
        best_val = 100  # arbitrary high initial value  

        # data collector
        losses_dict = {
            'train_loss': [],
            'validation_loss': [],
            'epoch': [],
            'best_val': []
        }

        # loop through epochs
        for epoch in range(start_epoch, config_dict['num_epochs']):

            epoch_begin = time.time()
            
            # train one epoch
            acc_train_loss, iter_count = train_one_epoch(train_loader, model, device, optimizer, 
                                                        config_dict['print_freq'], iter_count)

            # if applicable, lr schedule step
            if config_dict['lr_scheduler'] != "":
                lr_scheduler.step()
            
            # validate one epoch
            acc_val_loss = validate_one_epoch(validate_loader, model, device)

            # getting summariesed losses
            train_summary = sum(acc_train_loss) / len(acc_train_loss)
            validation_summary = sum(acc_val_loss) / len(acc_val_loss)
            #print("Training summary: %s, Validation summary: %s" % (train_summary, validation_summary))
            
            # save train data
            prev_best = best_val
            best_val = model_saver(epoch, model, optimizer, best_val, validation_summary,
                                    config_dict['out_dir'])
            
            losses_dict['train_loss'].append(train_summary)
            losses_dict['validation_loss'].append(validation_summary)
            losses_dict['epoch'].append(epoch)
            if best_val < prev_best:
                losses_dict['best_val'].append([epoch, best_val])

            delta = time.time() - epoch_begin
            epoch_duration = time_converter(delta) 
            print("Epoch Duration: ", epoch_duration)
            
        # recording losses    
        loss_path = config_dict['out_dir'] + "/loss_results.json"
        with open(loss_path, "w") as f:
            json.dump(losses_dict, f)

        
    # model evaluation
    #if config_dict['TEST']:
    #    evaluate(model, test_loader, device, config_dict['out_dir'])

    # producing plots    
    if config_dict['TRAIN']:
        print("plot and save train")
    if config_dict['TEST']:
        #print("plot and save test")
        
        # defining model locations 
        #losses_json = config_dict['out_dir'] + "/loss_results.json"
        #pr_json = config_dict['out_dir'] + "/precision_recall_results.json"
        
        # getting data from json
        #with open(losses_json) as losses_file:
        #    loss_dict = json.load(losses_file)
        #    losses_file.close() 
        #with open(pr_json) as pr_file:
        #    pr_dict = json.load(pr_file)
        #    pr_file.close()
        
        # plotting data
        #plot_lr_loss(loss_dict['epoch'], loss_dict['train_loss'], loss_dict['validation_loss'], 
        #             config_dict['plot_title'], config_dict['out_dir'])
        #plot_precision_recall(pr_dict['bbox'], config_dict['plot_title'], config_dict['out_dir'])
        
        #fps = fps_evaluate(model, config_dict['im_test_path'], device)
        #print(fps)
        
        segment_instance(device, config_dict['im_test_path'], ['__background__', 'jersey_royal'], model, config_dict['plot_title'], config_dict['out_dir'])

# ============================
# Train_net execution
# ============================
if __name__ == "__main__":
    
    # ========================
    # config dictionary
    # ========================

    conf_list = [configs.Mask_RCNN_R50_FPN_Base(), configs.Mask_RCNN_Mobilenet2_Base()]
    
    for conf in conf_list:

        # calling main    
        main(conf)