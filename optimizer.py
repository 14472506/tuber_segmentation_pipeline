"""
Details
"""
# imports
import torch

# finctions
def optimizer_selector(optimizer_title, model_params, optimizer_params):
    
    if optimizer_title == "SGD":
        optimizer = torch.optim.SGD(model_params, 
                                    lr = optimizer_params['lr'],
                                    momentum = optimizer_params['momentum'],
                                    weight_decay = optimizer_params['weight_decay'])

    return optimizer

def lr_scheduler_selector(scheduler_type, optimizer, step_size, gamma):
    """
    detials
    """
    if scheduler_type == "lr_step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    return(lr_scheduler)