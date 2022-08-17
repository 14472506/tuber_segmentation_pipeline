import numpy as np 

###################################################################################################
# classes                                                                            
###################################################################################################

class mAP_eval():
    """
    details
    """
    def __init__(self, ground_truth_data, predictions_data):
        """
        detials
        """
        self.ground_truth_data = ground_truth_data
        self.predictions_data = predictions_data
        self.confidences = []
        self.ious = []
        self.correct = []
        
        self.main()


    def main(self):
        """
        Detials
        """
        #get overlap matches
        gt_dict ,matches = self.compute_prediction_matches()

        # process matches
        self.process_matches_data(gt_dict, matches, 0.95)
        
        self.confidences
        self.ious
        self.correct
        prec_list = [0]*len(self.correct)
        rec_list = [0]*len(self.correct)
        results_table = np.array([self.confidences, self.correct, prec_list, rec_list])
        results_table = np.transpose(results_table)
        results_table = results_table[results_table[:, 0].argsort(kind='mergesort')[::-1]]

        precision, recall = self.compute_precision_recall(results_table, gt_dict)

        mAP = self.compute_mAP(precision, recall)
        print(mAP)


    def compute_prediction_matches(self):
        """
        Detials
        """
        # matches dict  structure = {gt_1: [p_1, overlap_1, ... , p_n, overlap_n],
        #                            ... ,
        #                            gt_l: [[p_1, overlap_1], ... , [p_n, overlap_n]]}
        matches = {}
        gt_dict = {}
        count = 0
        gt = self.ground_truth_data
        pred = self.predictions_data
        for i in range(len(gt)):
            for j in range(len(gt[i])):
                gt_dict[count] = gt[i][j]
                matches[count] = []
                for k in range(len(pred[i])):
                    overlap = np.count_nonzero(np.logical_and(gt[i][j]==1, pred[i][k]>0))
                    if overlap == 0:
                        continue
                    matches[count].append([pred[i][k], overlap])
                print(count)
                count += 1
        return(gt_dict, matches)


    def process_matches_data(self, gt_dict, matches, threshold):
        """
        Details
        """
        for key, val in matches.items():
            
            t_mask = gt_dict[key]
            ious = []
            confidences = []

            for i in val:
                p_mask = i[0]
                overlap = i[1]
                iou = self.compute_iou(p_mask, t_mask, overlap)
                if iou > threshold:
                    conf_val = np.min(p_mask[np.nonzero(p_mask)])
                    ious.append(iou)
                    confidences.append(conf_val)
            
            correct = [0]*len(ious)
            max_iou = max(ious)
            idx = ious.index(max_iou)
            correct[idx] = 1

            self.confidences.extend(confidences)
            self.ious.extend(ious)
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