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
from .models import model_selector
from .transforms import transform_selector
from .utils import make_dir
from .coco_evaluation import evaluate
from .data_loader import PipelineDataLoader

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
        #self.plot_title = configuration_dict['plot_title']
        #self.test_path = configuration_dict['im_test_path']
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
        self.optimizer = self.optimizer_selector()
        
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
            "validation_mAP": [],
            "best_model": [],
            "best_epoch": [],
            "model_params": None
        }
        self.main()

    # config methods
    # defining scheduling assistant
    def schedule_assigner(self):
        if self.scheduler_name != "":
            self.scheduler = self.lr_scheduler_selector()
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
        tr_load, v_load, _ = data_loader.manager()
        self.train_loader = tr_load
        self.val_loader = v_load

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
        self.training_data["model_params"] = int(params)
        

        # init best validation and iteration for loop
        iter_count = 0
        best_val = 0

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

            # getting map
            #torch.cuda.empty_cache()
            #mAP_val = evaluate(self.model, self.val_loader, self.device, self.out_dir,
            #                   train_flag=self.train)
            #mem_all = torch.cuda.memory_allocated(self.device) / 1024**3 
            #mem_res = torch.cuda.memory_reserved(self.device) / 1024**3 
            #mem = mem_res + mem_all
            #mem = round(mem, 2)
            #print("[mAP result][memory use: %sGB] mAP: %s" %(mem, mAP_val))

            #self.mAP_evaluation()
                    
            self.training_data["training_loss"].append(train_loss)
            self.training_data["validation_loss"].append(val_loss)
            #self.training_data["validation_mAP"].append(mAP_val)

            # saving models
            best_val = self.model_saver(epoch, best_val, val_loss, self.out_dir)
            
            # updating best val
            if best_val > val_loss:
                self.training_data["best_model"].append(val_loss)
                self.training_data["best_epoch"].append(epoch)
            
            # ending epoch timer and printing results
            epoch_end = time.time()
            delta = epoch_end - epoch_begin
            epoch_duration = self.time_converter(delta)
            print("Epoch Duration: ", epoch_duration)
        
        training_data = self.out_dir + "/training_data.json"
        with open(training_data, 'w') as f:
            json.dump(self.training_data, f)

        loss_path = self.out_dir + "/loss_results.json"
        with open(loss_path, "w") as f:
            json.dump(self.loss_dict, f)

    #############################################
    # Train one epoch
    #############################################

    # supporting methods to main
    # train one epoch
    def train_one_epoch(self, iter_count, epoch):
        """
        train details
        """
        # set/ensure model output is configured for trainingtrain_loader, model, device, optimizer,
        self.model.train()
        torch.cuda.empty_cache()

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

            # get GPU memory usage
            mem_all = torch.cuda.memory_allocated(self.device) / 1024**3 
            mem_res = torch.cuda.memory_reserved(self.device) / 1024**3 
            mem = mem_res + mem_all
            mem = round(mem, 2)

            # printing results 
            if iter_count % self.print_freq == 0:
                print("[epoch: %s][iter: %s][memory use: %sGB] total_loss: %s" %(epoch ,iter_count, mem, losses.item()))
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
        
        best_val = min(self.training_data["best_model"])
        idx = self.training_data["best_model"].index(best_val)
        epoch_val = self.training_data["best_epoch"][idx]
        
        #self.step_model["best_model"].append(best_model)
        #self.step_model["best_epoch"].append(best_epoch)

    # optimizer selector function
    def optimizer_selector(self):
        """
        Details
        """
        # for SGD optimizer
        if self.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(self.params, 
                                        lr = self.optimzier_parameters['lr'],
                                        momentum = self.optimzier_parameters['momentum'],
                                        weight_decay = self.optimzier_parameters['weight_decay'])

        # for ADAM optimizer
        if self.optimizer_name == "Adam":
            optimizer = torch.optim.Adam(self.params, lr = self.optimzier_parameters['lr'])

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

    #############################################
    # Validate one epoch
    #############################################
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
            torch.cuda.empty_cache()

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

        # get GPU memory usage
        mem_all = torch.cuda.memory_allocated(self.device) / 1024**3 
        mem_res = torch.cuda.memory_reserved(self.device) / 1024**3 
        mem = mem_res + mem_all
        mem = round(mem, 2)

        print("[validation_loss][memory use: %sGB] total_loss: %s" %(mem, sum(loss)/len(loss)))

        return(loss)
        
    def model_saver(self, epoch, best_result, mAP, path):
        """
        Details
        """
        # check if output dir exists, if not make dir
        make_dir(path)

        # making save dictionary for model.pth
        checkpoint = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        # saving last model
        last_model_path = path + "/last_model.pth"
        torch.save(checkpoint, last_model_path)

        if mAP > best_result:
            best_result = mAP
            best_model_path = path + "/best_model.pth"
            torch.save(checkpoint, best_model_path)

        return(best_result)

    def time_converter(self, seconds):
        """
        Detials
        """
        hours = seconds // 3600
        seconds %= 3600
        min = seconds // 60
        seconds %= 60

        return("%02d:%02d:%02d" %(hours, min, seconds))

    #############################################
    # mAP evaluation
    #############################################
    def mAP_evaluation(self):
        """
        Details
        """
        prediction_data = {}
        count = 0
        # setting model to eval 
        self.model.eval()    
        for images, targets in self.val_loader:

            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to("cpu") for k, v in t.items()} for t in targets]
                
            with torch.no_grad():
                torch.cuda.empty_cache()
                pred = self.model(images)

            pred = (pred[0]["masks"]).squeeze().detach().cpu().numpy()*1
            gt = (targets[0]["masks"]).detach().cpu().numpy()

            data = {
                "prediction": pred,
                "ground_truth": gt
            }
            prediction_data[count] = data

            print(count)

            count += 1

        #config_save = self.out_dir + "pred_dev.json"
        #with open(config_save, 'w') as f:
        #    json.dump(prediction_data, f)

        mAP_eval(prediction_data)

class mAP_eval():

    def __init__(self, prediction_data):
        """
        detials
        """
        self.prediction_data = prediction_data
        self.threshold = 0.05

        self.main()

    def main(self):
        """
        detials
        """
        for key, value in self.prediction_data.items():

            pred_masks = value["prediction"]
            targ_masks = value["ground_truth"]

            self.single_image_results(pred_masks, targ_masks)

    def single_image_results(self, pred_masks, targ_masks):
        """
        detials
        """
        all_ious = []
        iou_per_targ = []
        for t in targ_masks:
            t_ious = []
            for p in pred_masks:
                
                # getting overlap
                p = np.ceil(p)
                overlap = np.count_nonzero(np.logical_and( t==1,  p==1 ))

                # skipping if no overlap
                if overlap == 0:
                    continue

                p_area = np.count_nonzero(p == 1)
                t_area = np.count_nonzero(t == 1)

                iou = overlap/(p_area+t_area-overlap)

                if iou > self.threshold:
                    t_ious.append(iou)
                    all_ious.append(iou)

            iou_per_targ.append(t_ious)
        
        tp = len(all_ious)
        fp = pred_masks.shape[0] - tp
        fn = targ_masks.shape[0] - len(iou_per_targ)

        print(tp, fp, fn) 





                

            

            
    def get_AP(self, pred, targs, match_dict, threshold):
        """
        detials
        """
        # prediction and target masks
        pred_masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()*1
        targ_masks = targs[0]['masks'].detach().cpu().numpy()

        iou_list = []
        mask_idx = []
        pred_idx = []

        for key, val in match_dict.items():
            t_mask = targ_masks[key]
            for inst in val:
                p_mask = pred_masks[inst]
                p_mask = np.ceil(p_mask)

                iou = self.compute_iou(p_mask, t_mask)
                
                if iou > threshold:
                    iou_list.append(iou)
                    mask_idx.append(key)
                    pred_idx.append(inst)
        
        tp = len(iou_list)
        fp = pred_masks.shape[0] - tp
        fn = targ_masks.shape[0] - len([*set(mask_idx)])

        try:
            precision = tp/(tp + fp)
        except ZeroDivisionError:
            precision = 0.0
        try:
            recall = tp/(tp + fn)
        except ZeroDivisionError:
            recall = 0.0

        print(precision, recall)


    

