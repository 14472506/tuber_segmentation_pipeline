# something
"""
Detials
"""
# main imports
from tkinter import Image
import torch
import torchvision.transforms as T
from matplotlib import pyplot as plt
from models.models import model_selector
import numpy as np
import cv2
import random

# supporting imports
from PIL import Image as im

# suporting functions
def load_image(im_path):
    image = im.open(im_path)
    return(image)

#class
class Detector():
    
    def __init__(self, model_path):
        # supporting 
        self.im1 = "dev_folder/im1.jpg"
        self.im2 = "dev_folder/im2.jpg"
        self.im3 = "dev_folder/im3.jpg"

        # main 
        self.device = torch.device('cpu')
        self.model_path = model_path

        # execution
        self.load_model()
        self.predictor_imp()

    def load_model(self):
        """
        Detials
        """
        # calling the model architecture and inital config
        self.model = model_selector("Mask_RCNN_R50_FPN", 2, [800, 1333])
        checkpoint = torch.load(self.model_path)
        self.model.load_state_dict(checkpoint["state_dict"])
            
    def predictor_imp(self):
        
        pil_im = load_image(self.im3)
        
        transform = T.Compose([T.ToTensor()])
        img = transform(pil_im)
        img = img.to(self.device)

        # get prediction from model
        self.model.eval()
        with torch.no_grad():
            prediction = self.model([img])
        
        mask = self.best_mask(prediction, 0.5)
        uv = self.get_centroid(mask)

        print(uv)
        self.segment_instance(mask)
    
    def best_mask(self, pred, thresh):

        pred_masks = (pred[0]['masks']>thresh).squeeze().detach().cpu().numpy()*1
        pred_masks = pred_masks.astype(np.uint8)

        best_area = 0
        best_mask = None
        for pmask in pred_masks:
            
            area = np.count_nonzero(pmask == 1)
            #print(area)

            if area > best_area:
                best_area = area
                best_mask = pmask
        
        return(best_mask)

    def get_centroid(self, mask):
        """
        Details
        """
        # initialise the x and y points list
        points = np.where(mask == 1)
        x_points = points[1]
        y_points = points[0]

        x_points.sort()
        y_points.sort()

        x_idx = int((len(x_points) - 1)/2)
        y_idx = int((len(y_points) - 1)/2)

        centroid = [x_points[x_idx], y_points[y_idx]]

        # returning centroid
        return(centroid)

    def segment_instance(self, mask):
        
        pil_img = load_image(self.im3).convert('RGB')
        open_cv_image = np.array(pil_img) 
        img = open_cv_image[:, :, ::-1].copy()

        rgb_mask = self.get_coloured_mask(mask)
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        plt.figure(figsize=(20,30))
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def get_coloured_mask(self, mask):
        colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
        r = np.zeros_like(mask).astype(np.uint8)
        g = np.zeros_like(mask).astype(np.uint8)
        b = np.zeros_like(mask).astype(np.uint8)
        r[mask == 1], g[mask == 1], b[mask == 1] = colours[random.randrange(0,10)]
        coloured_mask = np.stack([r, g, b], axis=2)
        return coloured_mask

# execution
if __name__ == "__main__":
    model_path = "outputs/Saver_test_0.005/best_model.pth"
    #model_path = "dev_folder/R50_0.005.pth"
    detector = Detector(model_path)
