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
from models import model_selector
from transforms import transform_selector
from utils import make_dir
from coco_evaluation import evaluate
import config.configs as configs
from data_loader import PipelineDataLoader

# =================================================================================================
# train classes
# =================================================================================================
class TrainNet():
    """
    Detials
    """
    def __init__(self, configuration_dict, seed=42):
        """
        detials
        """
        # operation initialisation
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.seed = seed
        self.start_epoch = 0
        self.loss_dict = {}

        # initializing attributes from config dict
        self.out_dir = configuration_dict['out_dir']
        self.train = configuration_dict['TRAIN']
        self.test = False
        self.num_epochs = configuration_dict['num_epochs']
        self.print_freq = configuration_dict['print_freq']
        self.plot_title = configuration_dict['plot_title']
        self.test_path = configuration_dict['im_test_path']
        self.scheduler_params = configuration_dict['scheduler_params'] 
        
        # these need sorting at some point into areas
        self.model_name = configuration_dict['model'] 
        self.num_classes = configuration_dict['num_classes'] 
        self.max_min = configuration_dict['min_max']
        self.optimizer_name = configuration_dict['optimizer']
        self.optimzier_parameters = configuration_dict['optimizer_params']
        self.scheduler_name = configuration_dict['lr_scheduler'] 
        self.transforms_list = configuration_dict['transforms']
        self.batch_size = configuration_dict["batch_size"]
        self.loader_shuffle = configuration_dict["loader_shuffle"]
        self.loader_workers = configuration_dict["loader_workers"]
        self.train_dir = configuration_dict["train_dir"]
        self.train_json = configuration_dict["train_json"]
        self.val_dir = configuration_dict["val_dir"]
        self.val_json = configuration_dict["val_json"]
        self.load_flag = configuration_dict["load_flag"]
        self.load_checkpoint = configuration_dict["load"]

        # initializing method attributes
        self.model = model_selector(self.model_name, self.num_classes, 
                                    self.max_min) 
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = self.optimizer_selector(self.optimizer_name,
                                   self.params,
                                   self.optimzier_parameters)
        
        # calling methods  
        # condifuring transforms
        self.transforms_assigner()

        # configuring data loader
        self.loader_configuration_dict = {
            "seed": self.seed,
            "transforms": self.transforms, 
            "train": self.train,
            "test": self.test,
            "batch_size": self.batch_size, 
            "loader_shuffle": self.loader_shuffle,
            "loader_workers": self.loader_workers,
            "train_dir" : self.train_dir,
            "train_json" : self.train_json,
            "val_dir" : self.val_dir,
            "val_json" : self.val_json    
        }
        self.loader_assigner(self.loader_configuration_dict)

        # configuring optimizer
        self.optimizer_load()

        # configure scheduler        
        self.schedule_assigner()

        # saving training configuration file
        self.config_saver(configuration_dict)
        
        # training loop initialisiation
        self.training_data = {
            "training_loss": [],
            "validation_loss": [],
            "Validation_mAP": [],
            "best_model": [],
            "best_epoch": [],
            "model_params": None
        }
        self.main()

    # config methods
    # defining scheduling assistant
    def schedule_assigner(self):
        if self.scheduler_name != "":
            self.scheduler = self.lr_scheduler_selector(self.scheduler_name, self.optimizer,
                                                    self.scheduler_params)
        else:
            self.scheduler = None

    # defining transforms for data laoder
    def transforms_assigner(self):
        if self.transforms_list != "":
            self.transforms = transform_selector(self.transforms_list)
        else:
            self.transforms = None

    # defining data loader
    def loader_assigner(self, conf_dict):
        data_loader = PipelineDataLoader(conf_dict)
        tr_load, v_load, te_load = data_loader.manager()
        self.train_loader = tr_load
        self.val_loader = v_load
        self.test_loader = te_load
    
    # defining optimizer loader
    def optimizer_load(self):
        """
        Detials
        """
        if self.load_flag:
            checkpoint = torch.load(self.load_checkpoint)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.start_epoch = checkpoint["epoch"]
            for g in self.optimizer.param_groups:
                g['lr'] = self.optimzier_parameters['lr']

    # defining saver
    def config_saver(self, conf_dict):
        """
        Details
        """
        config_save = self.out_dir + "/model_configs.json"
        make_dir(self.out_dir)
        with open(config_save, 'w') as f:
            json.dump(conf_dict, f)

    # defining main training loop
    def main(self):
        """
        Detials
        """
        # send model to device
        self.model.to(self.device)

        # record parameters in model being trained
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.training_data["model_params"] = params

        # init best validation and iteration for loop
        iter_count = 0
        best_val = 100

        # training loop
        for epoch in range(self.start_epoch, self.num_epochs):
            # starting epoch timer
            epoch_begin = time.time()

            # train one epoch
            train_losses, iter_count = self.train_one_epoch(iter_count, epoch)

            # stepping scheduler if present
            if self.scheduler != None:
                # take scheduler step
                self.scheduler.step()
            
                # calling loop_loader logic
                self.loop_loader_logic(epoch)
            
            val_losses = self.validate_one_epoch()

            # loss from lossesprint_freq,
            train_loss = sum(train_losses)/len(train_losses)
            val_loss = sum(val_losses)/len(val_losses)

            self.training_data["training_loss"].append(train_loss)
            self.training_data["val_loss"].append(val_loss)

            # saving models
            self.model_saver(self, epoch, best_val, val_loss, self.out_dir)
            
            # updating best val
            if best_val > val_loss:
                self.training_data["best_model"].append(val_loss)
                self.training_data["best_epoch"].append(epoch)
            
            # ending epoch timer and printing results
            epoch_end = time.time()
            delta = epoch_end - epoch_begin
            epoch_duration = self.time_converter(delta)
            print("Epoch Duration: ", epoch_duration)

    # supporting methods to main
    # train one epoch
    def train_one_epoch(self, iter_count, epoch):
        """
        train details
        """
        # set/ensure model output is configured for trainingtrain_loader, model, device, optimizer,
        self.model.train()

        # loss_collection init
        loss = []

        # looping through dataset
        for images, targets in self.train_loader:

            # currently loading images and targets to device. REPLACE this with colate function and
            # and more effient method of sending images and target to device
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # passing batch to model to carry out forward
            # loss_dict contains loss tensors model losses. meta data such as loss functions and 
            # grad function also returned from model
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # recording loss data
            loss.append(losses.item())

            # carrying out backwards
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            # printing results 
            if iter_count % print_freq == 0:
                print("[epoch: %s][iter: %s] total_loss: %s" %(epoch ,iter_count, losses.item()))
            iter_count += 1    

        # return losses
        return loss, iter_count

    # loop logic loader
    def loop_loader_logic(self, epoch):
        """
        determins how best model is loaded as scheduler step based on scheduler type
        """
        if self.scheduler_name == "step":
            if epoch != 0:
                if epoch % self.scheduler_params[0] == 0:
                    self.loop_loader(epoch)

        if self.scheduler_name == "multi_step":
                if epoch in self.scheduler_params[0]:
                    self.loop_loader(epoch)

    # loop loader
    def loop_loader(self, epoch):
        """
        Detials
        """
        dir =  self.out_dir + "/best_model.pth"
        checkpoint = torch.load(dir)
        self.model.load_state_dict(checkpoint["state_dict"])
        
        mAP_val = max(self.best_model["mAP_val"])
        idx = self.best_model["mAP_val"].index(mAP_val)
        epoch_val = self.best_model["epoch"][idx]
        
        self.step_model["mAP_val"].append(mAP_val)
        self.step_model["epoch"].append(epoch_val)

    # optimizer selector function
    def optimizer_selector(self):
        """
        Details
        """
        # for SGD optimizer
        if self.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(self.params, 
                                        lr = self.optimizer_params['lr'],
                                        momentum = self.optimizer_params['momentum'],
                                        weight_decay = self.optimizer_params['weight_decay'])

        # for ADAM optimizer
        if self.optimizer_name == "Adam":
            optimizer = torch.optim.Adam(self.params, lr = self.optimizer_params['lr'])

        # return optimizer
        return optimizer
     
    # learning rate scheduler selector
    def lr_scheduler_selector(self):
        """
        detials
        """
        # step scheduling
        if self.scheduler_name == "step":
            lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                           step_size=self.scheduler_params[0],
                                                           gamma=self.scheduler_params[1])

        # multi step scheduling
        if self.scheduler_name == "multi_step":
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                milestones=self.scheduler_params[0],
                                                                gamma=self.scheduler_params[1])

        # returning scheduler
        return(lr_scheduler)

    # validate one epoch
    def validate_one_epoch(self):
        """
        validation details
        """
        # this is questionable and should be checked.
        self.model.train()

        # loss_collection
        loss = []

        # disables gradient calculation
        with torch.no_grad():

            # leaving this here for future reference for time being, all bn layers are frozen
            # therefor there should be no need to switch to eval

            # set the batch normalization in the model to eval
            #for module in model.modules():
            #    print(module)
                #if isinstance(module, torch.nn.BatchNorm2d):
                #    print(module)
                #if isinstance(module, torch.nn.BatchNorm2d):
                #    module.eval()

            """
            for name, module in module.name_modules():
              if hasattr(module, 'training'):
                print('{} is training {}'.format(name, module.training))
            """

            for images, targets in self.val_loader:

                # currently loading images and targets to device. REPLACE this with colate function and
                # and more effient method of sending images and target to device
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                # getting losses
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                # recording loss data
                loss.append(losses.item())

        return(loss)

    def model_saver(self, epoch, best_result, mAP, path):
        """
        Details
        """
        # check if output dir exists, if not make dir
        make_dir(path)

        # making save dictionary for model.pth
        checkpoint = {
            "epoch": epoch + 1,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        # saving last model
        last_model_path = path + "/last_model.pth"
        torch.save(checkpoint, last_model_path)

        if loss < best_result:
            #best_result = mAP
            best_model_path = path + "/best_model.pth"
            torch.save(checkpoint, best_model_path)

        #return(best_result)

    def time_converter(self, seconds):
        """
        Detials
        """
        hours = seconds // 3600
        seconds %= 3600
        min = seconds // 60
        seconds %= 60

        return("%02d:%02d:%02d" %(hours, min, seconds))