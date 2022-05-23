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
    if transfrom_string == "simple":
        transforms = simple_transforms()
    
def simple_transforms():
    """
    Details
    """
    # init transforms list
    transforms = []

    # adding random flip to list
    transforms.append(RandomHorizontalFlip(0.5))
    transforms.append(T.RandomGrayscale(0.1))
    transforms.append(T.RandomAutocontrast(0.1))

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
# THIS NEEDS WORK    
# randomly resizing the images within scale range 
class ScaleJitter(nn.Module):
    # " " "
    Randomly resizes the image and its bounding boxes  within the specified scale range.
    The class implements the Scale Jitter augmentation as described in the paper
    `"Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation" <https://arxiv.org/abs/2012.07177>`_.
    Args:
        target_size (tuple of ints): The target size for the transform provided in (height, weight) format.
        scale_range (tuple of ints): scaling factor interval, e.g (a, b), then scale is randomly sampled from the
            range a <= scale <= b.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
    # " " " 

    def __init__(
        self,
        target_size: Tuple[int, int],
        scale_range: Tuple[float, float] = (0.1, 2.0),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ):
        super().__init__()
        self.target_size = target_size
        self.scale_range = scale_range
        self.interpolation = interpolation

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(f"image should be 2/3 dimensional. Got {image.ndimension()} dimensions.")
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        _, orig_height, orig_width = F.get_dimensions(image)

        scale = self.scale_range[0] + torch.rand(1) * (self.scale_range[1] - self.scale_range[0])
        r = min(self.target_size[1] / orig_height, self.target_size[0] / orig_width) * scale
        new_width = int(orig_width * r)
        new_height = int(orig_height * r)

        image = F.resize(image, [new_height, new_width], interpolation=self.interpolation)

        if target is not None:
            target["boxes"][:, 0::2] *= new_width / orig_width
            target["boxes"][:, 1::2] *= new_height / orig_height
            if "masks" in target:
                target["masks"] = F.resize(
                    target["masks"], [new_height, new_width], interpolation=InterpolationMode.NEAREST
                )

        return image, target
"""
    
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