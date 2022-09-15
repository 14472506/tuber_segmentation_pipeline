import random
import os
import pickle as pkl
from re import sub

import torch
import numpy as np 

###################################################################################################
# classes                                                                            
###################################################################################################
class mAP_eval():
    """
    details
    """
    def __init__(self, ground_truth_data, predictions_data, prediction_boxes, ground_truth_boxes):
        """
        detials
        """
        self.ground_truth_data = ground_truth_data
        self.predictions_data = predictions_data
        self.ground_truth_boxes = ground_truth_boxes
        self.predictions_boxes = prediction_boxes

        self.confidences = []
        self.correct = []
        
        self.main()


    def main(self):
        """
        Detials
        """
        # get overlap matches
        gts ,matches, overlaps = self.compute_prediction_matches()

        # process matches
        ious, confidences = self.process_matches_data(gts, matches, overlaps)

        count = 0
        mAP_range = 0
        for thresh in range(5, 100, 5):
            
            thresh = thresh/100
            #print(thresh)
            self.iou_thresh_selection(ious, confidences, thresh)
            
            prec_list = [0]*len(self.correct)
            rec_list = [0]*len(self.correct)
            
            results_table = np.array([self.confidences, self.correct, prec_list, rec_list])
            results_table = np.transpose(results_table)
            results_table = results_table[results_table[:, 0].argsort(kind='mergesort')[::-1]]
            
            precision, recall = self.compute_precision_recall(results_table, gts)
            mAP = self.compute_mAP(precision, recall)
            print(mAP)

            if mAP > 0.09:
                count += 1
                mAP_range += mAP
            
            self.confidences = []
            self.correct = []

        print(mAP_range)
        print(mAP_range/count)

    def compute_prediction_matches(self):
        """
        Detials
        """
        # matches dict  structure = [[p_1, overlap_1, ... , p_n, overlap_n],
        #                            ... ,
        #                            [[p_1, overlap_1], ... , [p_n, overlap_n]]]
        matches = []
        gt_masks = []
        overlaps = []

        gt = self.ground_truth_data
        pred = self.predictions_data
        gt_boxes = self.ground_truth_boxes
        pred_boxes = self.predictions_boxes

        for tms_idx, t_masks in enumerate(gt): 
            p_masks = pred[tms_idx]
        
            for tm, t_mask in enumerate(t_masks):
                
                tb = gt_boxes[tms_idx][tm]
                tm_matches = []
                tm_overlaps = []
                gt_windows = []

                for pm, p_mask in enumerate(p_masks):

                    pb = pred_boxes[tms_idx][pm]

                    if tb[0] >= pb[2] or tb[2] <= pb[0]:
                        continue
                    if tb[1] >= pb[3] or tb[3] <= pb[1]:
                        continue
                    
                    overlap = 0

                    x_min = int(min(tb[0], pb[0])) 
                    y_min = int(min(tb[1], pb[1]))
                    x_max = int(max(tb[2], pb[2]))
                    y_max = int(max(tb[2], pb[3]))
                    
                    #print(x_min, x_max, y_min, y_max)
                    
                    t_mask_win = t_mask[y_min:y_max, x_min:x_max]
                    p_mask_win = p_mask[0][y_min:y_max, x_min:x_max]
                    
                    # PETRAS MODS, LOOK INTO THIS
                    # for x in range(max(tb[0], pb[0]), min(tb[2]), pb[2])):
                    # for x in range(tb[0], tb[2]):
                    #     for y in range(tb[1], tb[3]):
                    #         overlap += t_mask[y, x]==1 and p_mask[y, x] > 0

                    overlap = np.count_nonzero(np.logical_and(t_mask_win==1, p_mask_win>0))

                    if overlap == 0:
                        continue

                    tm_matches.append(p_mask_win)
                    gt_windows.append(t_mask_win)
                    tm_overlaps.append(overlap)

                matches.append(tm_matches)
                gt_masks.append(gt_windows)
                overlaps.append(tm_overlaps)

        return(gt_masks, matches, overlaps)
                    

    def process_matches_data(self, gts, matches, overlaps):
        """
        Details
        """
        iou_list = []
        confidence_list = []

        for i ,t_masks in enumerate(gts):
            match_masks = matches[i] 

            ious = []
            confidences = []
      
            for j, p_mask in enumerate(match_masks):

                t_mask = t_masks[j]
                overlap = overlaps[i][j]
                
                iou = self.compute_iou(p_mask, t_mask, overlap)
                
                if iou > .1:
                
                    ious.append(iou)

                    conf_val = np.min(p_mask[np.nonzero(p_mask)])
                    confidences.append(conf_val)
            
            iou_list.append(ious)
            confidence_list.append(confidences)
        
        return iou_list, confidence_list

    def iou_thresh_selection(self, iou_list, confidence_list, threshold):
        
        for i, ious in enumerate(iou_list):
        
            confidences = confidence_list[i]

            iou_set = []
            confidence_set = []

            for j, iou in enumerate(ious):
                if iou > threshold: 
                    iou_set.append(iou)
                    confidence_set.append(confidences[j])

            if len(iou_set) == 0:
                continue
            
            correct = [0]*len(iou_set)
            
            max_iou = max(iou_set)
            idx = iou_set.index(max_iou)

            correct[idx] = 1
            
            self.confidences.extend(confidence_set)
            self.correct.extend(correct)
                

    def compute_iou(self, pmask, tmask, overlap):
        """
        Details
        """
        p_area = np.count_nonzero(pmask>0)
        t_area = np.count_nonzero(tmask==1)
        iou = overlap/(p_area+t_area-overlap)
        return(iou)
    

    def compute_precision_recall(self, results_table, gt_dict):
        """
        Details
        """
        gt_count = len(gt_dict)
        tp_count = 0
        all_count = 0

        prec = []
        rec = []

        for row in results_table:
            tp_count += row[1]
            all_count += 1            
            prec.append(tp_count/all_count)
            rec.append(tp_count/gt_count)
                   
        return(prec, rec)

        
    def compute_mAP(self, prec, rec):
        """
        Details
        """
        prec_inter_col = {}

        prec_inter = []
        for i in range(len(prec)):
            try:
                prec_inter_col[rec[i]].append(prec[i])
            except KeyError:
                prec_inter_col[rec[i]] = [prec[i]]

        sm = 0
        rn = 0
        for key, val in prec_inter_col.items():
            sm += (key - rn)*max(val)
            print
            rn = key

        return(sm)

    def mAP_return(self):
        return(self.map)