"""
Details
"""
# ===========================
# Import libraties/packages
# ===========================
from cmath import nan
import torch
import numpy as np
from math import sqrt
import math as m
import sys
import json
import time
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import torchvision.transforms as T
import random

########
# for eval_forward
########
from collections import OrderedDict
from typing import Union
from torch import nn
import warnings
from torch.jit.annotations import Tuple, List, Dict, Optional
from torch import Tensor

# =================================================================================================
# Training
# =================================================================================================
def train_one_epoch(train_loader, model, device, optimizer, print_freq, iter_count, epoch):
    """
    train details
    """
    # set/ensure model output is configured for trainingtrain_loader, model, device, optimizer,
    model.train()

    # loss_collection
    loss_col = {
        'total': [],
        'classifier': [],
        'box_reg': [],
        'mask': [],
        'objectness': [],
        'rpn_box_reg': []
    }
    
    idx_list = ['classifier', 'box_reg', 'mask', 'objectness', 'rpn_box_reg']

    # looping through dataset
    for images, targets in train_loader:

        # currently loading images and targets to device. REPLACE this with colate function and
        # and more effient method of sending images and target to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # passing batch to model to carry out forward
        # loss_dict contains loss tensors model losses. meta data such as loss functions and 
        # grad function also returned from model
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # recording loss data
        loss_col['total'].append(losses.item())
        idx_count = 0
        for i in loss_dict.values():
            loss_col[idx_list[idx_count]].append(i.item())
            idx_count += 1
            
        # carrying out backwards
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # printing results 
        if iter_count % print_freq == 0:
            print("[epoch: %s][iter: %s] total_loss: %s" %(epoch ,iter_count, losses.item()))
        iter_count += 1    

    # return losses
    return loss_col, iter_count

# =================================================================================================
# Validation
# ================================================================================================= 
def validate_one_epoch(validation_loader, model, device):
    """
    validation details
    """
    # this is questionable and should be checked.
    model.train()

    # loss_collection
    loss_col = {
        'total': [],
        'classifier': [],
        'box_reg': [],
        'mask': [],
        'objectness': [],
        'rpn_box_reg': []
    }
    
    idx_list = ['classifier', 'box_reg', 'mask', 'objectness', 'rpn_box_reg'] 

    # disables gradient calculation
    with torch.no_grad():
        
        # leaving this here for future reference for time being, all bn layers are frozen
        # therefor there should be no need to switch to eval
    
        # set the batch normalization in the model to eval
        #for module in model.modules():
        #    print(module)
            #if isinstance(module, torch.nn.BatchNorm2d):
            #    print(module)
            #if isinstance(module, torch.nn.BatchNorm2d):
            #    module.eval()

        """
        for name, module in module.name_modules():
          if hasattr(module, 'training'):
            print('{} is training {}'.format(name, module.training))
        """
        
        for images, targets in validation_loader:

            # currently loading images and targets to device. REPLACE this with colate function and
            # and more effient method of sending images and target to device
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # getting losses
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # recording loss data
            loss_col['total'].append(losses.item())
            idx_count = 0
            for i in loss_dict.values():
                loss_col[idx_list[idx_count]].append(i.item())
                idx_count += 1
        
    return(loss_col)

# =================================================================================================
# Centroid Error Evaluation
# =================================================================================================
def centroid_evaluation(model, test_data_loader, device, output_dir):
    """
    Detials
    """
    # setting model to evaluate 
    model.eval()

    # intialising results dict
    results = {}

    with torch.no_grad():
        for thresh in range(50, 100, 5):
            # initialing thresh val and error list
            thresh_val = thresh/100
            centroid_errors = []

            # looping through test set
            for images, targets in test_data_loader:
                # sending images targets to gpu
                images = list(image.to(device) for image in images)
                #targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # predictions for model
                pred = model(images)

                # prediction and centroid lists
                p_cents, t_cents = centroid_lists(pred, targets, thresh_val)

                # append error list to centroid errors
                centroid_errors.append(get_errors(p_cents, t_cents)) 
        
            # adding results to results dictionary
            results[thresh] = centroid_errors
    
        file_name = output_dir + "/centroid_error.json"
        with open(file_name, "w") as f:
            json.dump(results, f)
    
def get_errors(pred_cents, targ_cents):
    """
    Details
    """
    # defining error list
    error_list = []

    # looping through prediction list
    for pred in pred_cents:
        # intialising arbitrary high error
        min_error = 1000

        # loop to check each target against prediction
        for targ in targ_cents:
            # getting error between target and prediction
            error = centroid_error(pred, targ)
            
            # checking and updating min error
            if error < min_error:
                min_error = error
        
        # appending smallest error to error list
        if min_error != 1000:
            error_list.append(min_error)
    
    # return error list
    return(error_list)

def centroid_error(pred_centroid, target_centroid):
    """
    Detials
    """
    # points for calculation
    x1 = pred_centroid[0]
    x2 = target_centroid[0]
    y1 = pred_centroid[1]
    y2 = target_centroid[1]
    
    # RMSE calculation
    error = round(abs(sqrt(abs((x1 - x2)**2 + abs(y1 - y2)**2))), 3) 

    # returning error
    return(error)

def centroid_lists(pred, targets, thresh):
    """
    Detials
    """
    # initialising centroid lists
    pred_centroids = []
    targ_centroids = []

    # getting mask lists
    pred_masks = (pred[0]['masks']>thresh).squeeze().detach().cpu().numpy()*1
    pred_masks = pred_masks.astype(np.uint8)
    targ_masks = targets[0]['masks'].detach().cpu().numpy()

    # get centroids for predictions and targets
    for pmask in pred_masks:
      pred_centroids.append(get_centroid(pmask)) #
    for tmask in targ_masks:
      targ_centroids.append(get_centroid(tmask))
    
    # returning collected prediction and centroid lists
    return pred_centroids, targ_centroids

def get_centroid(mask):
    """
    Details
    """
    # initialise the x and y points list
    x_points = []
    y_points = []

    # get rows and columns from mask
    rows = mask.shape[0]
    columns =  mask.shape[1]
        
    # get x and y list
    for i in range(columns):
        for j in range(rows):
            if mask[j, i] != 0:
                x_points.append(i)
                y_points.append(j)
                    
    # order lists
    x_points.sort()
    y_points.sort()
        
    # find centre indexes 
    x_idx = int((len(x_points)-1)/2)
    y_idx = int((len(y_points)-1)/2)

    # getting centroid
    centroid = [x_points[x_idx], y_points[y_idx]]

    # returning centroid
    return(centroid)

def get_centroid1(mask):
    """
    Detials
    """
    # get contours of mask for calculating image moments
    cnt, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # dealing with empty typles?
    if len(cnt) == 0:
        return [nan, nan]

    cnt = cnt[0]
    # calculate image moments
    M = cv2.moments(cnt)
    # using image moment calculate centroid x and y position
    if M['m00'] != 0: 
      cx = int(M['m10']/M['m00'])
      cy = int(M['m01']/M['m00'])
      # xy centroid point list
      point = [cx, cy]
    else:
      point = [nan, nan]
    
    # return centroid point
    return(point)

# ===== Centroid mask display =====================================================================
# =================================================================================================
def centroid_instance(device, img_path, test_loader, model, title, save_loc, thresh=0.5):
    """
    details
    """
    # set model to eval
    model.eval()
    
    # load evaluation image
    img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    img = img.to(device)
    
    # getting prediction from model
    with torch.no_grad():
        prediction = model([img])
    
    # getting masks from prediciton
    #pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    #pred_t = [pred_score.index(x) for x in pred_score if x>confidence][-1]
    pred_masks = (prediction[0]['masks']>thresh).squeeze().detach().cpu().numpy()
    pred_masks = pred_masks.astype(np.uint8)
    
    # getting image targets
    for image, target in test_loader:
        # selecting matching image target 
        if target[0]['image_id'].item() == 8:
            # collect mask
            target_masks = target[0]['masks'].detach().cpu().numpy()
            # break loop
            break
    
    # initialising prediction and target centroid lists
    pred_centroids = []
    targ_centroids = []
    # get centroids for predictions and targets
    for pmask in pred_masks:
      pred_centroids.append(get_centroid(pmask)) 
    for tmask in target_masks:
      targ_centroids.append(get_centroid(tmask))
    
    # getting matches
    match_list = get_matches(pred_centroids, targ_centroids)
    
    # plotting centroids
    plot_cents(img_path, match_list, title, save_loc)

def plot_cents(img_path, match_list, title, save_loc):
    """
    Detials
    """
    # defining colours for colour combinations
    #colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    
    # convert PIL image to numpy array
    img = cv2.imread(img_path)
    
    # loop through matches
    for match in match_list:
        # getting colour
        colour = [0, 255, 0]
        #colour = colours[random.randrange(0,10)
        
        if m.isnan(match[0][0]) == False and m.isnan(match[0][1]) == False:
        
            start = (match[0][0], match[0][1])
            end = (match[1][0], match[1][1])
            #print(match)
            # add match to image
            img = cv2.arrowedLine(img, start, end, colour, 2)
    
    # formatting image
    plt.figure(figsize=(20,30))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
  
    # saving 
    plot_string = save_loc + "/" + title + "_cent_error_fig.png"
    plt.savefig(plot_string)
     
def get_matches(pred_cents, targ_cents):
    """
    Details
    """
    # initialising best targets list
    best_targets = []
    
    # looping through prediction list
    for pred in pred_cents:
        # intialising arbitrary high error
        best_targ = [-1, -1]
        min_error = 1000

        # loop to check each target against prediction
        for targ in targ_cents:
            # getting error between target and prediction
            error = centroid_error(pred, targ)
            
            # checking and updating min error
            if error < min_error:
                best_targ = targ
                min_error = error
        
        # appending smallest error to error list
        if best_targ != [-1, -1]:
            best_targets.append(best_targ)
        else:
            best_targets.append([nan, nan])
    
    # creating list of tuples for line drawing iteration
    match_list = list(zip(pred_cents, best_targets))
    
    # return match list
    return(match_list)
    
    ##############################################
    # COLLECT MIN PRED TARG COMBINATIONS FOR PLOTTING
            
# =================================================================================================
# Evaluation
# ================================================================================================= 
# ===== For FPS evaluation =====
# ==============================
def fps_evaluate(model, image_path, device):
    """
    details

    Args:
        model (_type_): _description_
        image_path (_type_): _description_
        device (_type_): _description_
    """
    # loading image
    #model.eval()
    img = Image.open(image_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    img = img.to(device)
    
    # init times list
    times = []
    model.eval()
    with torch.no_grad():
      for i in range(10):
          start_time = time.time()

          model.eval()
          with torch.no_grad():
              pred = model([img])

          delta = time.time() - start_time
          times.append(delta)
    mean_delta = np.array(times).mean()
    fps = 1 / mean_delta
    return(fps)

# ===== for visualising segmnetation =====
# ========================================
def segment_instance(device, img_path, COCO_CLASS_NAMES, model, title, save_loc, confidence=0.5, rect_th=2, text_size=1, text_th=2):
    """
    segment_instance
      parameters:
        - img_path - path to input image
        - confidence- confidence to keep the prediction or not
        - rect_th - rect thickness
        - text_size
        - text_th - text thickness
      method:
        - prediction is obtained by get_prediction
        - each mask is given random color
        - each mask is added to the image in the ration 1:0.8 with opencv
        - final output is displayed
    """
    model.eval()
    masks, boxes, pred_cls = get_prediction(device, img_path, confidence, COCO_CLASS_NAMES, model)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(len(masks)):
        rgb_mask = get_coloured_mask(masks[i])
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        b1 = (int(boxes[i][0][0]), int(boxes[i][0][1]))
        b2 = (int(boxes[i][1][0]), int(boxes[i][1][1]))
        cv2.rectangle(img, b1, b2, color=(0, 255, 0), thickness=rect_th)
        cv2.putText(img,pred_cls[i], b1, cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    plt.figure(figsize=(20,30))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
  
    # saving 
    plot_string = save_loc + "/" + title + "_eval_img.png"
    plt.savefig(plot_string)
  
def get_coloured_mask(mask):
    """
    random_colour_masks
      parameters:
        - image - predicted masks
      method:
        - the masks of each predicted object is given random colour for visualization
    """
    colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask == 1], g[mask == 1], b[mask == 1] = colours[random.randrange(0,10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask

def get_prediction(device, img_path, confidence, COCO_CLASS_NAMES, model):
    """
    get_prediction
      parameters:
        - img_path - path of the input image
        - confidence - threshold to keep the prediction or not
      method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
        - masks, classes and bounding boxes are obtained from the model and soft masks are made binary(0 or 1) on masks
          ie: eg. segment of cat is made 1 and rest of the image is made 0

    """
    model.eval()
    img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    img = img.to(device)
    
    with torch.no_grad():
      pred = model([img])

    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x>confidence][-1]
    masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    pred_class = [COCO_CLASS_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    masks = masks[:pred_t+1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return masks, pred_boxes, pred_class