"""
Detials
"""
# =============================================================================================== #
# Import
# =============================================================================================== #
import torch

# =============================================================================================== #
# class
# =============================================================================================== #
class OptimizerConf():
    """
    Detials
    """
    def __init__(self, config_dict, model_params):
        """
        Detials
        """
        self.cd = config_dict
        self.opt_name = self.cd["OPTIMIZER"]["NAME"]
        self.opt_params = self.cd["OPTIMIZER"]["PARAMS"]
        self.sch_name = self.cd["SCHEDULER"]["NAME"]
        self.sch_params = self.cd["SCHEDULER"]["PARAMS"]
        self.model_params = model_params

        # get optimizer and scheduler
        self.optimizer_select()
        self.scheduler_select(self.optimizer)
        
    def optimizer_select(self):
        """
        Detials
        """
        if self.opt_name == "SGD":
            self.optimizer = torch.optim.SGD(self.model_params,
                                lr=self.opt_params[0],
                                momentum=self.opt_params[1],
                                weight_decay = self.opt_params[2])
        elif self.opt_name =="Adam":
            self.optimizer = torch.optim.Adam(self.model_params,
                                lr=self.opt_params[0])
        elif self.opt_name == "AdamW":
            self.optimizer = torch.optim.AdamW(self.model_params,
                                lr=self.opt_params[0])
        else:
            print("Optimizer not recognised")
    
    def scheduler_select(self, optimizer):
        """
        Details
        """
        if self.sch_name == "StepLR":
            self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                step_size=self.sch_params[0],
                                gamma=self.sch_params[1])
        elif self.sch_name == "MultiStepLR":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                milestones=self.sch_params[0],
                                gamma=self.sch_params[1])
        else:
            print("Scheduler not recognised")
    
    def optimizer_return(self):
        """
        Detials
        """
        return(self.optimizer)
    
    def scheduler_return(self):
        """
        Detials
        """
        return(self.scheduler)