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
import json

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
from utils import model_saver, make_dir, time_converter, set_seed, seed_worker
from coco_evaluation import evaluate
import configs
from plotter import plot_lr_loss, plot_precision_recall, plot_f1_score
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
    # ===== fixing random seed for reporducability ============================
    # =========================================================================
    # configuring device for cpu or gpu if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # fixing the random seed for reproducability
    set_seed(seed)
    
    # ===== handling data laodinging ==========================================
    # =========================================================================
    # configuring transforms for data loading
    if config_dict['transforms'] != "":
        transforms = transform_selector(config_dict['transforms'])
    else:
        transforms = None
    
    # producing generator with seed for data loader repeatability
    gen = torch.Generator()
    gen.manual_seed(seed)

    # get required datasets
    # TODO? could this be cleaner up?
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
    
    # ====== configuring model and solver =====================================
    # =========================================================================
    # get reqired model and set it to device
    model = model_selector(config_dict['model'], config_dict['num_classes'], 
                           config_dict['min_max'])
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
    if config_dict['load_flag']:
        checkpoint = torch.load(config_dict["load"])
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]

    # ====== saving config_dict ===============================================
    # =========================================================================
    # save train params here: ys save the json here!
    config_save = config_dict['out_dir'] + "/model_configs.json"
    make_dir(config_dict['out_dir'])
    with open(config_save, 'w') as f:
        json.dump(config_dict, f)

        
    # ===== entering training loops ===========================================
    # =========================================================================
    # training loop implementation
    if config_dict['TRAIN']: 
        # data trackers
        iter_count = 0
        best_val = 100  # arbitrary high initial value  

        # training loss dict
        train_loss = {
            'total': [],
            'classifier': [],
            'box_reg': [],
            'mask': [],
            'objectness': [],
            'rpn_box_reg': []
        }
        
        # validation loss dict
        val_loss = {
            'total': [],
            'classifier': [],
            'box_reg': [],
            'mask': [],
            'objectness': [],
            'rpn_box_reg': []
        }
        
        best_model = {
            'model_val': [],
            'epoch': []
        }
        
        # loop through epochs
        for epoch in range(start_epoch, config_dict['num_epochs']):
            
            # begin epoch count
            epoch_begin = time.time()
            
            # train one epoch
            acc_train_loss, iter_count = train_one_epoch(train_loader, model, device, optimizer, 
                                                        config_dict['print_freq'], iter_count, epoch)

            # if applicable, lr schedule step
            if config_dict['lr_scheduler'] != "":
                lr_scheduler.step()
            
            # validate one epoch
            acc_val_loss = validate_one_epoch(validate_loader, model, device)

            # getting summariesed losses
            train_summary = sum(acc_train_loss['total']) / len(acc_train_loss['total'])
            validation_summary = sum(acc_val_loss['total']) / len(acc_val_loss['total'])
            #print("Training summary: %s, Validation summary: %s" % (train_summary, validation_summary))
            
            # checking and saving model
            prev_best = best_val
            best_val = model_saver(epoch, model, optimizer, best_val, validation_summary,
                                    config_dict['out_dir'])
            
            # logging and saving results from training and validation
            train_loss['total'].append(train_summary)
            train_loss['classifier'].append(sum(acc_train_loss['classifier']) / len(acc_train_loss['classifier']))
            train_loss['box_reg'].append(sum(acc_train_loss['box_reg']) / len(acc_train_loss['box_reg']))
            train_loss['mask'].append(sum(acc_train_loss['mask']) / len(acc_train_loss['mask']))
            train_loss['objectness'].append(sum(acc_train_loss['objectness']) / len(acc_train_loss['objectness']))
            train_loss['rpn_box_reg'].append(sum(acc_train_loss['rpn_box_reg']) / len(acc_train_loss['rpn_box_reg']))
            
            val_loss['total'].append(validation_summary)
            val_loss['classifier'].append(sum(acc_val_loss['classifier']) / len(acc_val_loss['classifier']))
            val_loss['box_reg'].append(sum(acc_val_loss['box_reg']) / len(acc_val_loss['box_reg']))
            val_loss['mask'].append(sum(acc_val_loss['mask']) / len(acc_val_loss['mask']))
            val_loss['objectness'].append(sum(acc_val_loss['objectness']) / len(acc_val_loss['objectness']))
            val_loss['rpn_box_reg'].append(sum(acc_val_loss['rpn_box_reg']) / len(acc_val_loss['rpn_box_reg']))
            
            if best_val < prev_best:
                best_model['model_val'].append(best_val)
                best_model['epoch'].append(epoch+1)

            delta = time.time() - epoch_begin
            epoch_duration = time_converter(delta) 
            print("Epoch Duration: ", epoch_duration)
            
        # recording losses  
        losses_dict = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val': best_model
        }
        
        loss_path = config_dict['out_dir'] + "/loss_results.json"
        with open(loss_path, "w") as f:
            json.dump(losses_dict, f)

    # ===== calling evaluation and plotting ===================================
    # =========================================================================
    # producing plots    
    if config_dict['TRAIN']:
        #print("plot and save train")
        # path for loading json
        losses_json = config_dict['out_dir'] + "/loss_results.json"
        
        # getting data from json
        with open(losses_json) as losses_file:
            loss_dict = json.load(losses_file)
            losses_file.close()
        
        # producting lr loss plot
        plot_lr_loss(loss_dict, config_dict['plot_title'], config_dict['out_dir'])

    if config_dict['TEST']:
        # model evaluation
        evaluate(model, test_loader, device, config_dict['out_dir'])
        
        #print("plot and save test")
        # defining model locations 
        pr_json = config_dict['out_dir'] + "/precision_recall_results.json"
        
        # getting data from json
        with open(pr_json) as pr_file:
            pr_dict = json.load(pr_file)
            pr_file.close()
        
        # plotting data
        plot_precision_recall(pr_dict['segm'], config_dict['plot_title'], config_dict['out_dir'])
        #plot_f1_score(pr_dict['segm'], config_dict['plot_title'], config_dict['out_dir'])

        # fps_value
        fps = fps_evaluate(model, config_dict['im_test_path'], device)
        print(fps)
        
        # segmentation generation
        segment_instance(device, config_dict['im_test_path'], ['__background__', 'jersey_royal'], model, 
                         config_dict['plot_title'], config_dict['out_dir'])
    
# =================================================================================================
# Train_net execution
# =================================================================================================
if __name__ == "__main__":
    
    # conf dict list for experimental setup
    # Default setting as seen below:
    # ==============================
    # conf_maker(TRAIN, TEST, MODEL, OUT_DIR, TRANSFORMS="", LOAD_FLAG=False, LOAD_BEST=True, BATCH_SIZE=2,
    #            WORKERS=4 , MIN_MAX=[800, 1333], LR=0.005, NUM_EPOCHS=20,
    #            TEST_IM_STR="data/jersey_royal_dataset/test/169.JPG"):
    
    idx = 1
    lr_list = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005]
    
    for i in lr_list:
        
        # setting up list of models
        conf_list = [configs.conf_maker(True, False, "Mask_RCNN_R50_FPN", "TRANSFORM_TEST_"+str(idx), BATCH_SIZE=1,
                                        WORKERS=1, LR=i, NUM_EPOCHS=3, LOAD_FLAG=False, LOAD_BEST=False, 
                                        TRANSFORMS="random_flip")]

        # loop to train models through experiment
        for conf in conf_list:
            # calling main    
            main(conf)
        
        idx += 1