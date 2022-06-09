"""
Title:      models.py

Fuction:    The script handles all the models in the pipeline. with the models being handled by
            specific functions and the model_selector function returing the specified model. 

Edited by:  Bradley Hurst 
"""
# ============================
# importing libraries/packages
# ============================
# torch imports
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import MaskRCNN
from torch import nn

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

# ============================
# Model selector function
# ============================
# model selector
def model_selector(model_title, num_classes, min_max):
    """
    detials
    """
    if model_title == "Mask_RCNN_R50_FPN":
        get_model = ResnetSelector(num_classes, min_max, 'resnet50')
        model = get_model.return_model()
        
    if model_title == "Mask_RCNN_mobilenetv2":
        model = MaskRCNN_mobilenetv2(num_classes, min_max)
        
    if model_title == "Mask_RCNN_R18_FPN":
        get_model = ResnetSelector(num_classes, min_max, 'resnet18')
        model = get_model.return_model()
    
    if model_title == "Mask_RCNN_R34_FPN":
        get_model = ResnetSelector(num_classes, min_max, 'resnet34')
        model = get_model.return_model()
        
    if model_title == "test_selector":
        get_model = ResnetSelector(num_classes, min_max, 'resnet50')
        model = get_model.return_model()
    
    return(model)
    
# ============================
# Model call functions
# ============================
# Mask R-CNN, currently with resnet50 backbone
def MaskRCNN_R50_FPN_Basic(num_classes, min_max):
    # load an instance segmentation model pre-trained on COCO

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # setting min, max inpyt dimensions
    model.min_size = min_max[0]
    model.max_size = min_max[1]

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

def MaskRCNN_mobilenetv2(num_classes, min_max):
    # laoding pretrained backbone but returning only features
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features   

    # backbone output channels for mobilenetv2
    backbone.out_channels = 1280

    # defining anchor generator for model
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    # define roi features for roi cropping 
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)

    # define mask pooler for mask 
    mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                         output_size=14,
                                                         sampling_ratio=2)

    # model
    model = MaskRCNN(backbone,
                     num_classes=2,
                     min_size=min_max[0],
                     max_size=min_max[1],
                     rpn_anchor_generator=anchor_generator,
                     box_roi_pool=roi_pooler,
                     mask_roi_pool=mask_roi_pooler)
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


class ResnetSelector:
    """
    details
    """
    def __init__(self, num_classes, min_max, backbone, train_layers=5, trained=True):
        # getting attributes
        self.backbone = backbone = resnet_fpn_backbone(backbone, pretrained=trained,
                                                       trainable_layers=train_layers) 
        self.train_layers = train_layers
        self.num_classes = num_classes
        self.min_max = min_max
        
        # calling methods to get attributes
        self.get_model()
    
    def get_model(self):
        # getting anchor generator
        anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)),
        aspect_ratios=tuple([(0.25, 0.5, 1.0, 2.0) for _ in range(self.train_layers)]))

        # getting roi pooler
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],
                                                    output_size=7, sampling_ratio=2)

        # getting mask roi pooler 
        mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],
                                                         output_size=14,
                                                         sampling_ratio=2)

        # model
        self.model = MaskRCNN(self.backbone,
                         num_classes=self.num_classes,
                         min_size=self.min_max[0],
                         max_size=self.min_max[1],
                         rpn_anchor_generator=anchor_generator,
                         box_roi_pool=roi_pooler,
                         mask_roi_pool=mask_roi_pooler)
        
        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

        # now get the number of input features for the mask classifier
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
    
        # and replace the mask predictor with a new one
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                           hidden_layer,
                                                           self.num_classes)
        
    def return_model(self):
        return(self.model)