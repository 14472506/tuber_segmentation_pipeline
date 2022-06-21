"""
Details
"""
# imports
import torch

# functions
def optimizer_selector(optimizer_title, model_params, optimizer_params):
    """
    Details
    """
    if optimizer_title == "SGD":
        optimizer = torch.optim.SGD(model_params, 
                                    lr = optimizer_params['lr'],
                                    momentum = optimizer_params['momentum'],
                                    weight_decay = optimizer_params['weight_decay'])
    
    if optimizer_title == "Adam":
        optimizer = torch.optim.Adam(model_params, lr = optimizer_params['lr'])

    return optimizer

def lr_scheduler_selector(scheduler_type, optimizer, scheduler_params):
    """
    detials
    """
    # step scheduling
    if scheduler_type == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_params[0],
                                                        gamma=scheduler_params[1])
    
    # multi step scheduling
    if scheduler_type == "multi_step":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=scheduler_params[0],
                                                        gamma=scheduler_params[1])

    # returning scheduler
    return(lr_scheduler)