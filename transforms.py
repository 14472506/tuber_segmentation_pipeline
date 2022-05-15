"""
Details
"""
import torchvision.transforms as T

def transform_selector(transforms_str):
    if transforms_str == "simple":
        transforms = simple_transform()
    
    return transforms

def simple_transform():
    """
    
    """
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor

    # during training, randomly flip the training images
    # and ground-truth for data augmentation
    #transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.RandomGrayscale(0.25))
    transforms.append(T.RandomAutocontrast(0.25))
    
    return T.Compose(transforms)