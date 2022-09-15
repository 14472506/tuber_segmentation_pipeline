"""
Detaisl
"""
# imports
import numpy as np

# class
class mAP_eval():
    """
    Details
    """
    def __init__(self, predictions_data):
        """
        Details
        """
        self.predictions_data = predictions_data
        self.threshold_lower = 5
        self.threshold_upper = 95
        self.all_mAPs = []
        self.mAPs = []

        self.main()

    def main(self):
        """
        detials
        """
        # add loop for iterating over iou thresh.
        for i in range(self.threshold_lower, self.threshold_upper, 5):

            # initialising list for collecting true postatibes and treu and false negatives
            tps = []
            fns = []
            fps = []

            # initialising list for precision and recall
            precisions = []
            recalls = []

            # getting threshold value
            thresh = i/100

            # looping through 
            for key, value in self.predictions_data.items():
                
                # getting masks
                pred_masks = value["prediction"]
                targ_masks = value["ground_truth"]

                # calculating true posatives and false posatives and negatives from image
                tp, fp, fn = self.single_image_results(pred_masks, targ_masks, thresh)

                # collecting true posativesand false postatives and negatives
                tps.append(tp)
                fns.append(fn)
                fps.append(fp)
                
                # calculating precision and recall
                prec, rec = self.precision_recall(tps, fps, fns)

                # collecting precision and recall 
                precisions.append(prec)
                recalls.append(rec)

            # get mAP val for precision and recall
            mAP_val = self.get_mAP(precisions, recalls)

            self.mAPs.append(mAP_val)

        self.mAP = np.mean(self.mAPs)

    def get_mAP(self, precisions, recalls):
        """
        detials
        """
        
        # convert to np array
        precisions = np.array(precisions)
        recalls = np.array(recalls)
        
        # init list for collecting precision at recall
        prec_at_rec = []

        # looping ovar all precision values
        for recall_level in np.linspace(0.0, 1.0, 11):
            try:
                idx = np.argwhere(recalls >= recall_level).flatten()
                prec = max(precisions[idx])
            except ValueError:
                prec = 0.0
            prec_at_rec.append(prec)
        
        # calcultating mAP
        mAP_val = np.mean(prec_at_rec)
        
        return(mAP_val)

    def single_image_results(self, pred_masks, targ_masks, thresh):
        """
        detials
        """
        all_ious = []
        iou_per_targ = []
        for t in targ_masks:
            t_ious = []

            for p in pred_masks:
                # getting overlap
                p = np.ceil(p)
                overlap = np.count_nonzero(np.logical_and( t==1,  p==1 ))
                # skipping if no overlap
                if overlap == 0:
                    continue

                p_area = np.count_nonzero(p == 1)
                t_area = np.count_nonzero(t == 1)

                iou = overlap/(p_area+t_area-overlap)

                if iou >= thresh:
                    t_ious.append(iou)
                    all_ious.append(iou)

            iou_per_targ.append(t_ious)

        tp = len(all_ious) 
        fp = pred_masks.shape[0] - tp
        fn = targ_masks.shape[0] - len(iou_per_targ)

        return tp, fp, fn

    def precision_recall(self, tps, fps, fns):
        """
        detials
        """
        try:
            precision = sum(tps)/(sum(tps) + sum(fps))
        except ZeroDivisionError:
            precision = 0.0
        try:
            recall = sum(tps)/(sum(tps) + sum(fns))
        except ZeroDivisionError:
            recall = 0.0
        
        return precision, recall

    def mAP_call(self):
        """
        detials
        """
        return self.mAP

# main
if __name__ == "__main__":

    a = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    
    b = np.transpose(a)

    c = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    
    d = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                  [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                  [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    
    dict = {
        1: {"prediction": np.array([a]), "ground_truth": np.array([d])},
        2: {"prediction": np.array([a]), "ground_truth": np.array([d])},
        3: {"prediction": np.array([a]), "ground_truth": np.array([d])},
        4: {"prediction": np.array([a]), "ground_truth": np.array([d])},
    }

    large_dict = {}
    for i in range(15):
        
        p  = []
        t = []

        count = 0

        for j in range(100):

            if count == 0:
                p.append(a)
            else:
                p.append(b)
        
        t.append(a)
        
        p = np.array(p)
        t = np.array(t)

        large_dict[i] = {"prediction": p, "ground_truth": t}
    
    test = mAP_eval(large_dict)
    mAP = test.mAP_call()
    print(mAP)