"""
Details
"""
# =================================================================================================
# importing libraries and functions
# =================================================================================================
from turtle import width
from typing import Optional, Tuple, Dict, List

import torch 
import torchvision

from torch import Tensor, nn
from torchvision.transforms import transforms as T
from torchvision.transforms import functional as F

# =================================================================================================
# Transform selector
# =================================================================================================
def transform_selector(transfrom_string):
    if transfrom_string == "shape_transforms":
        transforms = shape_transforms()
        
    if transfrom_string == "colour_transforms":
        transforms = colour_transforms()
        
    if transfrom_string == "combine_transforms":
        transforms = combine_transforms()
    
def shape_transforms():
    """
    Details
    """
    # init transforms list
    transforms = []

    # adding random flip to list
    transforms.append(RandomHorizontalFlip(0.5))
    transforms.append(RandomIoUCrop())    
    
    # retuning composed transform list
    return(Compose(transforms))

def colour_transforms():
    """
    Details
    """
    # init transforms list
    transforms = []

    # adding random flip to list    
    transforms.append(T.RandomGrayscale(0.2))
    transforms.append(T.RandomAutocontrast(0.2))
    
    # retuning composed transform list
    return(Compose(transforms))

def combine_transforms():
    """
    Details
    """
    # init transforms list
    transforms = []

    # adding random flip to list
    transforms.append(RandomHorizontalFlip(0.5))
    transforms.append(RandomIoUCrop())    
    transforms.append(T.RandomGrayscale(0.2))
    transforms.append(T.RandomAutocontrast(0.2))
    
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

class RandomIoUCrop(nn.Module):
    def __init__(
        self,
        min_scale: float = 0.3,
        max_scale: float = 1.0,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2.0,
        sampler_options: Optional[List[float]] = None,
        trials: int = 40,
    ):
        super().__init__()
        # Configuration similar to https://github.com/weiliu89/caffe/blob/ssd/examples/ssd/ssd_coco.py#L89-L174
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        if sampler_options is None:
            sampler_options = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        self.options = sampler_options
        self.trials = trials

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if target is None:
            raise ValueError("The targets can't be None for this transform.")

        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(f"image should be 2/3 dimensional. Got {image.ndimension()} dimensions.")
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        _, orig_h, orig_w = F.get_dimensions(image)

        while True:
            # sample an option
            idx = int(torch.randint(low=0, high=len(self.options), size=(1,)))
            min_jaccard_overlap = self.options[idx]
            if min_jaccard_overlap >= 1.0:  # a value larger than 1 encodes the leave as-is option
                return image, target

            for _ in range(self.trials):
                # check the aspect ratio limitations
                r = self.min_scale + (self.max_scale - self.min_scale) * torch.rand(2)
                new_w = int(orig_w * r[0])
                new_h = int(orig_h * r[1])
                aspect_ratio = new_w / new_h
                if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                    continue

                # check for 0 area crops
                r = torch.rand(2)
                left = int((orig_w - new_w) * r[0])
                top = int((orig_h - new_h) * r[1])
                right = left + new_w
                bottom = top + new_h
                if left == right or top == bottom:
                    continue

                # check for any valid boxes with centers within the crop area
                cx = 0.5 * (target["boxes"][:, 0] + target["boxes"][:, 2])
                cy = 0.5 * (target["boxes"][:, 1] + target["boxes"][:, 3])
                is_within_crop_area = (left < cx) & (cx < right) & (top < cy) & (cy < bottom)
                if not is_within_crop_area.any():
                    continue

                # check at least 1 box with jaccard limitations
                boxes = target["boxes"][is_within_crop_area]
                ious = torchvision.ops.boxes.box_iou(
                    boxes, torch.tensor([[left, top, right, bottom]], dtype=boxes.dtype, device=boxes.device)
                )
                if ious.max() < min_jaccard_overlap:
                    continue

                # keep only valid boxes and perform cropping
                target["boxes"] = boxes
                target["labels"] = target["labels"][is_within_crop_area]
                target["boxes"][:, 0::2] -= left
                target["boxes"][:, 1::2] -= top
                target["boxes"][:, 0::2].clamp_(min=0, max=new_w)
                target["boxes"][:, 1::2].clamp_(min=0, max=new_h)
                
                # cropping images and masks
                image = F.crop(image, top, left, new_h, new_w)
                target["masks"] = F.crop(target["masks"], top, left, new_h, new_w)

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