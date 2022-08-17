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
        self.plot_title = conf_dict['plot_title']
        self.test_path = conf_dict['im_test_path']
        self.scheduler_params = conf_dict['scheduler_params']
        self.training_data = {
            'train_total': [],
            'train_classifier': [],
            'train_box_reg': [],
            'train_mask': [],
            'train_objectness': [],
            'train_rpn_box_reg': [],
            'val_total': [],
            'val_classifier': [],
            'val_box_reg': [],
            'val_mask': [],
            'val_objectness': [],
            'val_rpn_box_reg': [],
            'val_mAP': [],
            'best_mAP': [],
            'step_mAP': [],
            'best_epoch': [],
            'step_epoch': [],
            'parameters': None
        }

        # calling attribute assigning methods  
        self.loader_assigner(conf_dict)
        self.optimizer_load(conf_dict)
        self.schedule_assigner(conf_dict)

        # save exp_configuration
        config_save = self.out_dir + "/model_configs.json"
        make_dir(self.out_dir)
        with open(config_save, 'w') as f:
            json.dump(conf_dict, f)
        
        # training loop initialisiation
        self.main()
    
    # =========================================================================================== #
    # ----- Init function calls ----------------------------------------------------------------- #
    # =========================================================================================== # 
    
    def loader_assigner(self, conf_dict):
        if conf_dict['transforms'] != "":
            self.transforms = transform_selector(conf_dict['transforms'])
        else:
            self.transforms = None
        tr_load, v_load, te_load = data_loader_manager(conf_dict, self.seed, self.transforms)
        self.train_loader = tr_load
        self.val_loader = v_load
        self.test_loader = te_load
    
    # something
    def optimizer_load(self, conf_dict):
        if conf_dict['load_flag']:
            checkpoint = torch.load(conf_dict["load"])
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.start_epoch = checkpoint["epoch"]
            for g in self.optimizer.param_groups:
                g['lr'] = conf_dict['optimizer_params']['lr']
    
    # config methods
    def schedule_assigner(self, conf_dict):
        if conf_dict['lr_scheduler'] != "":
            self.scheduler = lr_scheduler_selector(conf_dict['lr_scheduler'], self.optimizer,
                                                    self.scheduler_params)
            self.scheduler_title = conf_dict['lr_scheduler']
        else:
            self.scheduler = None
            self.scheduler_title = None
    
    def main(self):
        """
        details
        """
        # sending model to device
        self.model.to(self.device)
        
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.training_data['parameters'] = int(params)

        # executing training 
        if self.train:
            self.training_exe()

        # executing testing
        if self.test:
            self.testing_exe()

    # =========================================================================================== #
    # ----- Supporting functions for main ------------------------------------------------------- #
    # =========================================================================================== #
    def loop_loader_logic(self, epoch):
        """
        determins how best model is loaded as scheduler step based on scheduler type
        """
        if self.scheduler_title == "step":
            if epoch != 0:
                if epoch % self.scheduler_params[0] == 0:
                    self.loop_loader(epoch)

        if self.scheduler_title == "multi_step":
                if epoch in self.scheduler_params[0]:
                    self.loop_loader(epoch)
        
    def loop_loader(self, epoch):
        """
        Detials
        """
        dir =  self.out_dir + "/best_model.pth"
        checkpoint = torch.load(dir)
        self.model.load_state_dict(checkpoint["state_dict"])
        
        mAP_val = max(self.training_data["val_mAP"])
        idx = self.training_data["best_mAP"].index(mAP_val)
        epoch_val = self.training_data["best_epoch"][idx]
        
        self.training_data["setp_mAP"].append(mAP_val)
        self.training_data["step_epoch"].append(epoch_val)

    def training_exe(self):
        iter_count = 0
        best_val = 100 # changed for loss

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

                # calling loop_loader logic
                self.loop_loader_logic(epoch)
            
            # validating one epoch
            acc_val_loss = validate_one_epoch(self.val_loader, self.model, self.device)#

            # validation eval
            #mAP_val = evaluate(self.model, self.val_loader, self.device, self.out_dir,
            #                    train_flag=self.train)

            # collecting data
            self.training_data['train_total'].append(sum(acc_train_loss['total']) / len(acc_train_loss['total']))
            self.training_data['val_total'].append(sum(acc_val_loss['total']) / len(acc_val_loss['total']))
            #self.training_data['val_mAP'].append(mAP_val)
            val_total = sum(acc_val_loss['total'])/len(acc_val_loss['total'])
            # saving models
            prev_best = best_val
            best_val = model_saver(epoch, self.model, self.optimizer, best_val, val_total, self.out_dir) # modified model_saver
            #if best_val > prev_best:
            #    self.training_data['best_mAP'].append(best_val)
            #    self.training_data['best_epoch'].append(epoch+1)
            if best_val < prev_best:
                self.training_data['best_mAP'].append(best_val)
                self.training_data['best_epoch'].append(epoch+1)
            
            # finish epoch count
            delta = time.time() - epoch_begin
            epoch_duration = time_converter(delta)
            print("Epoch Duration: ", epoch_duration)
        
        # saving data in json
        save_file = self.out_dir + "/training_data.json"
        with open(save_file, "w") as f:
            json.dump(self.training_data, f)

    def testing_exe(self):
        # carrying out evaluations
        evaluate(self.model, self.test_loader, self.device, self.out_dir, test_flag=self.test)
        #centroid_evaluation(self.model, self.test_loader, self.device, self.out_dir)

        # loading json data
        #pr_json = self.out_dir + "/precision_recall_results.json"
        #cen_err_json = self.out_dir + "/centroid_error.json"
        #with open(pr_json) as pr_file:
        #    pr_dict = json.load(pr_file)
        #    pr_file.close()
        #with open(cen_err_json) as ce_file:
        #    ce_dict = json.load(ce_file)
        #    ce_file.close()

        # plotting data
        #plot_precision_recall(pr_dict['segm'], self.plot_title, self.out_dir)
        #plot_cent_err(ce_dict, self.plot_title, self.out_dir)
        #plot_f1_score(pr_dict['segm'], self.plot_title, self.out_dir)

        # fps_value
        fps = fps_evaluate(self.model, self.test_path, self.device)
        print(fps)
        
        # segmentation generation
        #segment_instance(self.device, self.test_path, ['__background__', 'jersey_royal'], self.model, 
        #                 self.plot_title, self.out_dir)
        #centroid_instance(self.device, self.test_path, self.test_loader, self.model, self.plot_title,
        # 