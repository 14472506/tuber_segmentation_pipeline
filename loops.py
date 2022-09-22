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
import json

# ----- pytorch imports
import torch

# ----- package imports
from utils import make_dir, model_saver
from dataloader import COCOLoader, collate_function
from model import ModelSelector
from optimizer import OptimizerConf
from coco_evaluation import evaluate
from mAP_eval import mAP_eval
from transforms import transform_selector

# =============================================================================================== #
# classes
# =============================================================================================== #
# ----- Training Loop --------------------------------------------------------------------------- #
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

        # setting seed
        self.set_seed()

        # loading datasets
        self.train_root = self.cd["DATASET"]["TRAIN"]
        self.train_trans = transform_selector(self.cd["DATASET"]["TRAIN_TRANSFORMS"])
        self.val_root = self.cd["DATASET"]["VAL"]
        self.train_trans = transform_selector(self.cd["DATASET"]["VAL_TRANSFORMS"])
        self.load_dataset()

        # get model and send it to device
        m = ModelSelector(config_dict)
        self.model = m.return_model()
        self.model.to(self.device)
        self.params = [p for p in self.model.parameters() if p.requires_grad]
    
        # optimizer and schedule config
        o = OptimizerConf(config_dict,
                self.params)
        self.optimizer = o.optimizer_return()
        self.scheduler = o.scheduler_return()

        # loop config
        self.save_config()
        self.recording_init()
        self.start_epoch = self.cd["LOOP"]["START_EPOCH"]
        self.num_epochs = self.cd["LOOP"]["END_EPOCH"]
        self.print_freq = self.cd["REPORTING"]["PRINT_FREQ"]
        self.loop()
        
    
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
            self.train_loader = torch.utils.data.DataLoader(
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
            self.val_loader = torch.utils.data.DataLoader(
                            val_data,
                            batch_size = self.cd['DATALOADER']['VAL']['BATCH_SIZE'],
                            shuffle = self.cd['DATALOADER']['VAL']['SHUFFLE'],
                            num_workers = self.cd['DATALOADER']['VAL']['WORKERS'],
                            worker_init_fn = seed_worker,
                            generator = gen,
                            collate_fn = collate_function)
        else:
            print("No Validation Dataset Loaded")


    def recording_init(self):
        """
        Details
        """
        self.data_recording = {
            'train_total'       : [],
            'train_classifier'  : [],
            'train_box_reg'     : [],
            'train_mask'        : [],
            'train_objectness'  : [],
            'train_rpn_box_reg' : [],
            'val_total'         : [],
            'val_classifier'    : [],
            'val_box_reg'       : [],
            'val_mask'          : [],
            'val_objectness'    : [],
            'val_rpn_box_reg'   : [],
            'val_mAP'           : [],
            'best_mAP'          : [],
            'step_mAP'          : [],
            'best_epoch'        : [],
            'step_epoch'        : [],
            'parameters'        : None
        }
    
    
    def save_config(self):
        """
        Detials
        """
        # saving data in json
        save_file = self.exp_dir + "/config.json"
        with open(save_file, "w") as f:
            json.dump(self.cd, f)

    def loop(self):
        """
        Detials
        """
        # collecting model parameters
        model_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.data_recording['parameters'] = int(sum([np.prod(p.size()) for p in model_params]))

        # config best_model selector
        best_model = 100
 
        iter_count = 0
        for epoch in range(self.start_epoch, self.num_epochs, 1):
            # run training on one epoch
            epoch_training_loss, iter_count = self.train_one_epoch(epoch, iter_count)

            # run validation on one epoch
            epoch_val_loss = self.val_one_epoch(epoch)

            # save last model
            model_saver(epoch, self.model, self.optimizer, self.exp_dir, "last_model.pth")

            # save best model
            if best_model >= epoch_val_loss:
                best_model = epoch_val_loss
                model_saver(epoch, self.model, self.optimizer, self.exp_dir, "best_model.pth")
                self.data_recording["best_mAP"].append(best_model)
                self.data_recording["best_epoch"].append(epoch)
                

            # scheduler management
            if self.scheduler != None:    
                
                # scheduler step
                self.scheduler.step() 

                # check logic
                best_model = self.scheduler_step_logic(epoch, best_model)

            
        # saving data in json
        save_file = self.exp_dir + "/training_data.json"
        with open(save_file, "w") as f:
            json.dump(self.data_recording, f)


    def scheduler_step_logic(self, epoch, best_val):
        """
        Detials
        """
        if self.cd["SCHEDULER"]["NAME"] == "StepLR":
            if epoch != 0:
                if epoch % self.cd["SCHEDULER"]["PARAMS"][0] == 0:
                    self.schedule_loader(epoch, best_val)
                    return 100
                else:
                    return best_val 
            return best_val

        if self.cd["SCHEDULER"]["NAME"] == "MultiStepLR":
            if epoch in self.cd["OPTIMIZER"]["PARAMS"][0]:
                self.schedule_loader(epoch, best_val)
                return 100
            else:
                return best_val 


    def schedule_loader(self, epoch, best_val):
        """
        Detials
        """
        # load best model so far
        model_dir = self.exp_dir + "/best_model.pth"
        checkpoint = torch.load(model_dir)
        self.model.load_state_dict(checkpoint["state_dict"])

        # save best model as best pre step model
        last_model_path = self.exp_dir + "/ps_best_model.pth"
        torch.save(checkpoint, last_model_path)

        # best pre_step val results
        val_res = min(self.data_recording["val_total"])
        idx = self.data_recording["best_mAP"].index(val_res)
        epoch_val = self.data_recording["best_epoch"][idx]
        self.data_recording["step_mAP"].append(val_res)
        self.data_recording["step_epoch"].append(epoch_val)


    def train_one_epoch(self, epoch, iter_count):
        """
        Detials
        """
        # set model to train
        self.model.train()

        # initialising accumulators
        total_acc = 0
        class_acc = 0
        boxre_acc = 0  
        masks_acc = 0
        objec_acc = 0 
        rpnbr_acc = 0 

        # entering epoch loop
        for images, targets in self.train_loader:
            
            # send images and targets to device
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # forward + backward + optimizer
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
           
            # setting param gradient to zero
            self.optimizer.zero_grad()
           
            losses.backward()
            self.optimizer.step()

            # recording loss data
            total_acc += losses.item()
            class_acc += loss_dict["loss_classifier"].item()
            boxre_acc += loss_dict["loss_box_reg"].item() 
            masks_acc += loss_dict["loss_mask"].item()
            objec_acc += loss_dict["loss_objectness"].item()
            rpnbr_acc += loss_dict["loss_rpn_box_reg"].item()

            # results printing
            if iter_count % self.print_freq == 0: #self.print_freq-1:
                
                # get GPU memory usage
                mem_all = torch.cuda.memory_allocated(self.device) / 1024**3 
                mem_res = torch.cuda.memory_reserved(self.device) / 1024**3 
                mem = mem_res + mem_all
                mem = round(mem, 2)
                print("[epoch: %s][iter: %s][memory use: %sGB] total_loss: %s" %(epoch ,iter_count, mem, losses.item()))

            # add one to iter count
            iter_count += 1
        
        # recording data
        self.data_recording["train_total"].append(total_acc/len(self.train_loader)) 
        self.data_recording["train_classifier"].append(class_acc/len(self.train_loader))
        self.data_recording["train_box_reg"].append(boxre_acc/len(self.train_loader))
        self.data_recording["train_mask"].append(masks_acc/len(self.train_loader))
        self.data_recording["train_objectness"].append(objec_acc/len(self.train_loader))
        self.data_recording["train_rpn_box_reg"].append(rpnbr_acc/len(self.train_loader))
        
        # returning results
        return total_acc/len(self.train_loader), iter_count


    def val_one_epoch(self, epoch):
        """
        Detials
        """
        # set model to train: Note this is due to match val and tain losses
        self.model.train()

        # initialising accumulators
        total_acc = 0
        class_acc = 0
        boxre_acc = 0  
        masks_acc = 0
        objec_acc = 0 
        rpnbr_acc = 0 

        # with torch.no_grad() so no gradients are calculated for validation
        with torch.no_grad():
            # entering epoch loop
            for images, targets in self.val_loader:
            
                # load images and targets
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                # getting losses (this serves as a prediction)
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                # recording loss data
                total_acc += losses.item()
                class_acc += loss_dict["loss_classifier"].item()
                boxre_acc += loss_dict["loss_box_reg"].item() 
                masks_acc += loss_dict["loss_mask"].item()
                objec_acc += loss_dict["loss_objectness"].item()
                rpnbr_acc += loss_dict["loss_rpn_box_reg"].item()

        # recording data
        self.data_recording["val_total"].append(total_acc/len(self.val_loader)) 
        self.data_recording["val_classifier"].append(class_acc/len(self.val_loader))
        self.data_recording["val_box_reg"].append(boxre_acc/len(self.val_loader))
        self.data_recording["val_mask"].append(masks_acc/len(self.val_loader))
        self.data_recording["val_objectness"].append(objec_acc/len(self.val_loader))
        self.data_recording["val_rpn_box_reg"].append(rpnbr_acc/len(self.val_loader))  
        
        # get GPU memory usage
        mem_all = torch.cuda.memory_allocated(self.device) / 1024**3 
        mem_res = torch.cuda.memory_reserved(self.device) / 1024**3 
        mem = mem_res + mem_all
        mem = round(mem, 2)
        print("[epoch: %s][iter: val][memory use: %sGB] total_loss: %s" %(epoch, mem, total_acc/len(self.val_loader)))

        return total_acc/len(self.val_loader)

# ----- Evaluation Loop ------------------------------------------------------------------------- #
class EvalLoop():
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

        # set random seed:
        self.set_seed()

        # data location
        self.test_root = self.cd["DATASET"]["TEST"]
        self.load_dataset()

        # model location
        m = ModelSelector(config_dict)
        self.model = m.return_model()
        self.load_model()
        self.model.to(self.device)

        # evaluate
        self.eval_loop()

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

        # load training dataset
        test_data = COCOLoader(
                        root = self.test_root, 
                        json_root = self.test_root + self.cd["DATASET"]["TEST_JSON"], # put back comma when augmenting 
                        transforms = self.cd['DATASET']["TEST_TRANSFORMS"]
                        )
        
        # configuring train loader
        self.test_loader = torch.utils.data.DataLoader(
                        test_data,
                        batch_size = self.cd['DATALOADER']['TEST']['BATCH_SIZE'],
                        shuffle = self.cd['DATALOADER']['TEST']['SHUFFLE'],
                        num_workers = self.cd['DATALOADER']['TEST']['WORKERS'],
                        worker_init_fn = seed_worker,
                        generator = gen,
                        collate_fn = collate_function)


    def load_model(self):
        """
        Detials
        """
        # load best model so far
        model_dir = self.exp_dir + "/ps_best_model.pth"
        checkpoint = torch.load(model_dir)
        self.model.load_state_dict(checkpoint["state_dict"])


    def eval_loop(self):
        """
        Detials
        """
        mAP = evaluate(self.model, self.test_loader, self.device, self.exp_dir, train_flag=True)
        print(mAP)