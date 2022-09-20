"""
Detials
"""
# =============================================================================================== #
# imports
# =============================================================================================== #
# ----- python imports
import random
import numpy as np
import os

# ----- pytorch imports
import torch

# ----- package imports
from utils import make_dir
from dataloader import COCOLoader, collate_function
from model import ModelSelector
from optimizer import OptimizerConf

# =============================================================================================== #
# classes
# =============================================================================================== #
class TrainLoop():
    """
    Detials
    """
    def __init__(self, config_dict, seed=42):
        """
        Detials
        """
        # getting config_dictionary
        self.cd = config_dict

        # setups
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.seed = seed

        # experimental data setup
        self.experiment_name = self.cd['EXPERIMENT_NAME']
        self.exp_dir = "outputs/" + self.experiment_name
        make_dir(self.exp_dir)

        # data location
        self.train_root = self.cd["DATASET"]["TRAIN"]
        self.val_root = self.cd["DATASET"]["VAL"]

        # setting seed
        self.set_seed()

        # loading datasets
        self.load_dataset()

        # get model and send it to device
        m = ModelSelector(config_dict)
        self.model = m.return_model()
        self.model.to(self.device)
    
        # optimizer and schedule config
        o = Optimizer(self.model.opt_parameters(),
                config_dict)
        self.optimizer = o.optimizer_return()
        self.scheduler = o.scheduler_return()

        # loop config
        self.recording_init()
        self.start_epoch = self.cd["LOOP"]["START_EPOCH"]
        self.num_epochs = self.cd["LOOP"]["END_EPOCH"]
        self.loop():
        
    
    def set_seed(self):
        """
        Details
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(self.seed)

    
    def load_dataset(self, seed=42):
        """
        Detials
        """
        # defining internal seed worker function
        def seed_worker(worker_id):
            """
            Details
            """
            info = torch.utils.data.get_worker_info()
            worker_seed =  torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed) 
            random.seed(worker_seed)


        # producing generator for seed setting in data loader
        gen = torch.Generator()
        gen.manual_seed(seed)

        # loading training and validation datasets
        if self.train_root:
            # load training dataset
            train_data = COCOLoader(
                            root = self.train_root, 
                            json_root = self.train_root + self.cd["DATASET"]["TRAIN_JSON"], # put back comma when augmenting 
                            transforms = self.cd['DATASET']["TRAIN_TRANSFORMS"]
                            )
            
            # configuring train loader
            train_loader = torch.utils.data.DataLoader(
                            train_data,
                            batch_size = self.cd['DATALOADER']['TRAIN']['BATCH_SIZE'],
                            shuffle = self.cd['DATALOADER']['TRAIN']['SHUFFLE'],
                            num_workers = self.cd['DATALOADER']['TRAIN']['WORKERS'],
                            worker_init_fn = seed_worker,
                            generator = gen,
                            collate_fn = collate_function)
        else:
            print("No Training Dataset Loaded")
        
        if self.val_root:
            val_data = COCOLoader(
                            root = self.val_root, 
                            json_root = self.val_root + self.cd["DATASET"]["VAL_JSON"], # put back comma when augmenting 
                            transforms = self.cd['DATASET']["VAL_TRANSFORMS"]
                            )
            
            # configuring val loader
            val_loader = torch.utils.data.DataLoader(
                            train_data,
                            batch_size = self.cd['DATALOADER']['VAL']['BATCH_SIZE'],
                            shuffle = self.cd['DATALOADER']['VAL']['SHUFFLE'],
                            num_workers = self.cd['DATALOADER']['VAL']['WORKERS'],
                            worker_init_fn = seed_worker,
                            generator = gen,
                            collate_fn = collate_function)
        else:
            print("No Validation Dataset Loaded")
    

    def reporting_init(self):
        """
        Details
        """
        self.data_recording = {
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
    

    def loop(self):
        """
        Detials
        """
        # collecting model parameters
        model_params = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prob(p.size()) for p in model_parameters])
        self.data_recording['parameters'] = int(params)

        # config best_model selector
        best_model = 100

        for epoch in range(self.start_epoch, self.num_epochs, 1):
            # run training on one epoch
            epoch_training_loss = self.train_one_epoch(epoch)

            # run validation on one epoch
            epoch_val_loss = self.val_one_epoch()

            # save last model
            model_saver(epoch, self.model, self.optimizer, self.exp_dir, "las_model.pth")

            # save best model
            if epoch_val_loss < best_model:
                model_saver(epoch, self.model, self.optimizer, self.exp_dir, "best_model.pth")
    

    def train_one_epoch(self, epoch):
        """
        Detials
        """
        # set model to train
        self.model.train()

        # entering epoch loop
        for images, targets in self.train_loader:
            # send images and targets to device
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # setting param gradient to zero
            self.optimizer.zero_grad

            # forward + backward + optimizer
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # recording loss data
            self.data_recording["total"].append()


    
    def val_one_epoch(self, epoch):
        """
        Detials
        """
        pass