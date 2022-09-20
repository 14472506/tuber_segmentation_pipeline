"""
Detials
"""
# =============================================================================================== #
# Imports
# =============================================================================================== #
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import MaskRCNN
from torch import nn

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

# =============================================================================================== #
# Model selector
# =============================================================================================== #
class ModelSelector():
    """
    Details
    """
    def __init__(self, config_dict):
        """
        Details
        """
        self.cd = config_dict
        self.model_name = self.cd["MODEL"]["MODEL_NAME"]
        self.backbone_name = self.cd["MODEL"]["BACKBONE_NAME"]
        self.b_type = self.backbone_name.split("_")[0]

        # get backbone
        self.backbone_selector()

        # get model
        self.get_model()

    def get_model(self):
        """
        Detials
        """
        # getting anchor generator
        #anchor_generator = AnchorGenerator(
        #sizes=((16,), (32,), (64,), (128,), (256,)),
        #aspect_ratios=tuple([(0.25, 0.5, 1.0, 2.0) for _ in range(self.train_layers)]))

        # getting roi pooler
        #roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],
        #                                            output_size=7, sampling_ratio=2)

        # getting mask roi pooler 
        #mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],
        #                                                 output_size=14,
        #                                                 sampling_ratio=2)

        # model
        self.model = MaskRCNN(self.backbone,
                         num_classes=self.cd["MODEL"]["NUM_CLASSES"],
                         min_size=self.cd["MODEL"]["MIN"],
                         max_size=self.cd["MODEL"]["MAX"],
        #                 rpn_anchor_generator=anchor_generator,
        #                 box_roi_pool=roi_pooler,
        #                 mask_roi_pool=mask_roi_pooler
        )
        
        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=self.cd["MODEL"]["NUM_CLASSES"])

        # now get the number of input features for the mask classifier
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
    
        # and replace the mask predictor with a new one
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                hidden_layer,
                                                num_classes=self.cd["MODEL"]["NUM_CLASSES"])
    
    def backbone_selector(self):
        """
        Detials
        """
        if self.b_type == "resnet":
            r_type = self.backbone_name.split("_")[1]
            resnet = self.b_type + r_type
            self.resnet_selector(resnet)
        elif self.b_type == "mobilenet":
            pass
        else:
            print("Backbone not recognised")
    
    def resnet_selector(self, resnet):
        """
        Detail
        """
        self.backbone = resnet_fpn_backbone(resnet,
                            pretrained = self.cd["MODEL"]["TRAINED"],
                            trainable_layers = self.cd["MODEL"]["TRAINABLE_LAYERS"])

    def return_model(self):
        """
        Detials
        """
        return(self.model)