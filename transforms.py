"""
Details
"""
# =================================================================================================
# importing libraries and functions
# =================================================================================================
from turtle import width
from typing import Optional, Tuple, Dict

import torch 
import torchvision

from torch import Tensor, nn
from torchvision.transforms import transforms as T
from torchvision.transforms import functional as F

# =================================================================================================
# Transform selector
# =================================================================================================
def transform_selector(transfrom_string):
    if transfrom_string == "random_flip":
        transforms = random_flip()
    
def random_flip():
    """
    Details
    """
    # init transforms list
    transforms = []

    # adding random flip to list
    transforms.append(RandomHorizontalFlip(0.5))

    # retuning composed transform list
    return(Compose(transforms))

# =================================================================================================
# Transforms
# =================================================================================================
# Compose class called for returning transformed images
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

# Horizontal flip implementation for image and tensor
class RandomHorizontalFlip(T.RandomHorizontalFlip):
    # definig modified forward for Torch.RandomHorizontalFlip
    def forward(self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
                ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        
        # implementing probability of implementation activation
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            
            # if target
            if target is not None:
                _, _, width = F.get_dimensions(image)
                # flipping boxes in target
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]

                # flipping masks in target
                if "masks" in target:
                    target["masks"] = target["masks"].flip(-1)

        return image, target


    



"""
############################################################## OLD STUFF, FUNCTIONS for now
def transform_selector(transforms_str):
    if transforms_str == "simple":
        transforms = simple_transform()
    
    return transforms

def simple_transform():transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor

    # during training, randomly flip the training images
    # and ground-truth for data augmentation
    #transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.RandomGrayscale(0.25))
    transforms.append(T.RandomAutocontrast(0.25))
    
    return T.Compose(transforms)
"""