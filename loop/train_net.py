"""
Title:      train_network.py

Fuction:    This script acts as the top level script for training all models for potato tuber
            segmentation. The script will configure the pipeline to the required parameters for
            the specific model being trained. If the script runs successful, the pipeline will
            produce a trained model

Edited by:  Bradley Hurst 
"""
# =================================================================================================
# Importing packages/libraries 
# =================================================================================================
# pytorch imports
import json
import torch 
import torchvision.transforms as T

# suporting library imports
import numpy as np
import random
import time

# package imports
from models.models import model_selector
from .optimizer import optimizer_selector, lr_scheduler_selector
from .engine import train_one_epoch, validate_one_epoch, fps_evaluate, segment_instance, centroid_error, centroid_instance
from data.transforms import transform_selector
from .utils import model_saver, make_dir, time_converter, set_seed, data_loader_manager
from evaluation.coco_evaluation import evaluate
import config.configs as configs
from evaluation.plotter import plot_lr_loss#, plot_precision_recall, plot_f1_score, plot_cent_err
# =================================================================================================
# Train_net implementation
# =================================================================================================
class TrainNetwork:
    """
    class detials
    """
    def __init__(self, conf_dict, seed=42):
        """
        details
        """
        # collecting  configuration attributes
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.seed = seed
        self.model = model_selector(conf_dict['model'], conf_dict['num_classes'], 
                                    conf_dict['min_max']) 
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optimizer_selector(conf_dict['optimizer'],
                                   self.params,
                                   conf_dict['optimizer_params'])
        self.start_epoch = 0
        self.out_dir = conf_dict['out_dir']
        self.train = conf_dict['TRAIN']
        self.test = conf_dict['TEST']
        self.num_epochs = conf_dict['num_epochs']
        self.print_freq = conf_dict['print_freq']
        self.loss_dict = {}
        self.plot_title = conf_dict['plot_title']
        self.test_path = conf_dict['im_test_path']
        self.scheduler_params = conf_dict['scheduler_params']

        # calling attribute assigning methods  
        self.transforms_assigner(conf_dict)
        self.loader_assigner(conf_dict)
        self.optimizer_load(conf_dict)
        self.schedule_assigner(conf_dict)
        self.config_saver(conf_dict)
        
        # training loop initialisiation
        self.data_logging_init()
        self.main()
    
    # config methods
    def schedule_assigner(self, conf_dict):
        if conf_dict['lr_scheduler'] != "":
            self.scheduler = lr_scheduler_selector(conf_dict['lr_scheduler'], self.optimizer,
                                                    self.scheduler_params)
        else:
            self.scheduler = None

    def transforms_assigner(self, conf_dict):
        if conf_dict['transforms'] != "":
            self.transforms = transform_selector(conf_dict['transforms'])
        else:
            self.transforms = None

    def loader_assigner(self, conf_dict):
        tr_load, v_load, te_load = data_loader_manager(conf_dict, self.seed, self.transforms)
        self.train_loader = tr_load
        self.val_loader = v_load
        self.test_loader = te_load
    
    def optimizer_load(self, conf_dict):
        if conf_dict['load_flag']:
            checkpoint = torch.load(conf_dict["load"])
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.start_epoch = checkpoint["epoch"]
            for g in self.optimizer.param_groups:
                g['lr'] = conf_dict['optimizer_params']['lr']
    
    def config_saver(self, conf_dict):
        config_save = self.out_dir + "/model_configs.json"
        make_dir(self.out_dir)
        with open(config_save, 'w') as f:
            json.dump(conf_dict, f)
    
    def data_logging_init(self):
        if self.train: 

            # training loss dict
            self.train_loss = {
                'total': [],
                #'classifier': [],
                #'box_reg': [],
                #'mask': [],
                #'objectness': [],
                #'rpn_box_reg': []
            }
        
            # validation loss dict
            self.val_loss = {
                'total': [],
                #'classifier': [],
                #'box_reg': [],
                #'mask': [],
                #'objectness': [],
                #'rpn_box_reg': []
            }
        
            # evaluation dicitionary
            self.val_eval = {
                'mAP': []
            }
        
            self.best_model = {
                'mAP_val': [],
                'epoch': []
            }

    def main(self):
        """
        details
        """
        # sending model to device
        self.model.to(self.device)
        
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(params)
        
        # executing training 
        if self.train:
            self.training_exe()

        # executing testing
        if self.test:
            self.testing_exe()
    
    def training_exe(self):
        iter_count = 0
        best_val = 0

        # loop through the epochs
        for epoch in range(self.start_epoch, self.num_epochs):
            # begin epoch count
            epoch_begin = time.time()

            # train one epoch
            acc_train_loss, iter_count = train_one_epoch(self.train_loader, self.model, self.device,
                                                self.optimizer, self.print_freq, iter_count, epoch)
            
            # stepping scheduler if present
            if self.scheduler != None:
                self.scheduler.step()

                if epoch != 0:
                    if epoch % self.scheduler_params[0] == 0:
                        dir =  self.out_dir + "/best_model.pth"
                        checkpoint = torch.load(dir)
                        self.model.load_state_dict(checkpoint["state_dict"])
            
            # validating one epoch
            acc_val_loss = validate_one_epoch(self.val_loader, self.model, self.device)#

            # validation eval
            mAP_val = evaluate(self.model, self.val_loader, self.device, self.out_dir,
                                train_flag=self.train)

            # collecting data
            self.train_loss['total'].append(sum(acc_train_loss['total']) / len(acc_train_loss['total']))
            self.val_loss['total'].append(sum(acc_val_loss['total']) / len(acc_val_loss['total']))
            self.val_eval['mAP'].append(mAP_val)

            # saving models
            prev_best = best_val
            best_val = model_saver(epoch, self.model, self.optimizer, best_val, mAP_val, self.out_dir)
            if best_val > prev_best:
                self.best_model['mAP_val'].append(best_val)
                self.best_model['epoch'].append(epoch+1)
            
            # finish epoch count
            delta = time.time() - epoch_begin
            epoch_duration = time_converter(delta)
            print("Epoch Duration: ", epoch_duration)

        # recording losses  
        self.loss_dict = {
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'val_eval': self.val_eval,
            'best_val': self.best_model
        }
        
        # saving data in json
        loss_path = self.out_dir + "/loss_results.json"
        with open(loss_path, "w") as f:
            json.dump(self.loss_dict, f)

        # plotting data
        plot_lr_loss(self.loss_dict, self.plot_title, self.out_dir)

    def testing_exe(self):
        # carrying out evaluations
        evaluate(self.model, self.test_loader, self.device, self.out_dir, test_flag=self.test)
        #centroid_evaluation(self.model, self.test_loader, self.device, self.out_dir)

        # loading json data
        pr_json = self.out_dir + "/precision_recall_results.json"
        cen_err_json = self.out_dir + "/centroid_error.json"
        with open(pr_json) as pr_file:
            pr_dict = json.load(pr_file)
            pr_file.close()
        with open(cen_err_json) as ce_file:
            ce_dict = json.load(ce_file)
            ce_file.close()

        # plotting data
        #plot_precision_recall(pr_dict['segm'], self.plot_title, self.out_dir)
        #plot_cent_err(ce_dict, self.plot_title, self.out_dir)
        #plot_f1_score(pr_dict['segm'], self.plot_title, self.out_dir)

        # fps_value
        fps = fps_evaluate(self.model, self.test_path, self.device)
        print(fps)
        
        # segmentation generation
        segment_instance(self.device, self.test_path, ['__background__', 'jersey_royal'], self.model, 
                         self.plot_title, self.out_dir)
        centroid_instance(self.device, self.test_path, self.test_loader, self.model, self.plot_title,
                         self.out_dir, thresh=0.95)