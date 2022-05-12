"""
Title:      dataloader.py

Fuction:    The script is the location for all elements of loading datasets into the
            tuber segmentation pipeline. this includes the functions used to structure the datsets
            for use in the model and for functions that assist that load the data into the models
            for use. 

Edited by:  Bradley Hurst 
"""
# ============================
# Importing libraries/packages
# ============================  
# torch imports
import json
import torch
import torchvision.transforms as T
import torch.utils.data as data

# supporting package imports
import os 
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from PIL import Image

# ============================
# Classes and functions for data loading
# ============================
class COCOLoader(data.Dataset):
    """
    Title:      COCOLoader

    Function:   The class inherits from the torch.utils.data.Dataset and modifies the __getitem__  
                and __len__ methods for the function 

    Inputs:     - json string
                - image dir string 
                - transforms
                
    Outputs:    - a image tensor and target tensor when __getitem__ method is called
                - a id length value when __rep__ is called

    Deps:

    Edited by:  Bradley Hurst
    """
    def __init__(self, root, json_root, transforms=None):
        self.root = root
        self.coco = COCO(json_root)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __getitem__(self, idx):
        
        # getting ids for specific image
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds = img_id)
        
        # loading annotations for annotation ids
        anns = self.coco.loadAnns(ann_ids)

        # initialisng lists from target data
        labels = []
        boxes = []        
        masks_list = []
        areas = []
        iscrowds = []
        
        # itterating though loaded anns
        for ann in anns:
            
            # collecting data labels and areas for target
            labels.append(ann['category_id'])
            areas.append(ann['area'])

            # formatting and collecting bbox data 
            bbox = ann['bbox']            
            new_bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
            boxes.append(new_bbox)

            # formatting and collecting iscrowd id's
            if ann["iscrowd"]:
                iscrowds.append(1)
            else:
                iscrowds.append(0)

            # formatting mask to tensor and collecting masks
            mask = self.coco.annToMask(ann)
            mask == ann['category_id']
            masks_list.append(torch.from_numpy(mask))

        # converting lists to tensors for target
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = torch.as_tensor(areas, dtype=torch.int64)
        masks = torch.stack(masks_list, 0)
        iscrowd = torch.as_tensor(iscrowds, dtype=torch.int64)
        image_id = torch.tensor([idx])

        # assembling target dict
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        # laoding image
        image_path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, image_path)).convert('RGB')
        
        # converting to tensor
        im_conv = T.ToTensor()
        img = im_conv(img)

        # applying transforms if applicable
        if self.transforms != None:
            img, target = self.transforms(img, target)

        return img, target
    
    def __len__(self):
        return len(self.ids)

def collate_function(batch):
    """
    Title:      collate_finction

    Function:   function formats how batch is formated

    Inputs:     batch
                
    Outputs:    formatted batch

    Deps:

    Edited by:  Bradley Hurst
    """
    return tuple(zip(*batch))

    ## before testing, all this will need to be tested when on the web
    #images = torch.stack(images, 0)

# ============================
# For format conversion
# ============================
def convert_to_coco_api(ds):
    """
    details
    """
    # initialise coco requirments
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()

    # iterate through dataset
    for img_idx in range(len(ds)):
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        # get image and target in image idx
        img, targets = ds[img_idx]

        # getting image id from target
        image_id = targets["image_id"].item()
        
        # initialise and assemble image dict
        img_dict = {}
        img_dict["id"] = image_id
        img_dict["height"] = img.shape[-2]
        img_dict["width"] = img.shape[-1]
        
        # ad image dict to dataset dict
        dataset["images"].append(img_dict)
        
        # collect bounding box data
        bboxes = targets["boxes"].clone()
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        
        # collecting label, area, and iscrowd data
        labels = targets["labels"].tolist()
        areas = targets["area"].tolist()
        iscrowd = targets["iscrowd"].tolist()

        # collecting masks
        masks = targets["masks"]
        # make masks Fortran contiguous for coco_mask
        masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)

        # constucting anns
        num_objs = len(bboxes)
        for i in range(num_objs):
            # collecting annotation content in loop
            ann = {}
            ann["image_id"] = image_id
            ann["bbox"] = bboxes[i]
            ann["category_id"] = labels[i]
            categories.add(labels[i])
            ann["area"] = areas[i]
            ann["iscrowd"] = iscrowd[i]
            ann["id"] = ann_id
            if "masks" in targets:
                ann["segmentation"] = coco_mask.encode(masks[i].numpy())
            
            # appending annotations to dataset
            dataset["annotations"].append(ann)
            ann_id += 1
    
    # complete assembling coco dataset dict
    dataset["categories"] = [{"id": i} for i in sorted(categories)]

    # formating dataset dict with 
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds