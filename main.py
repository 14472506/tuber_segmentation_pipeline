"""
Details
"""
# imports
from loop.train_test import TrainNetwork
import config.configs as configs

# main function
def main():

    # Boolean flags
    TRAIN = False
    TEST = True
    LOAD = True
    BEST = True
    
    # othere params
    EPOCHS = 150
    BATCH_SIZE = 1
    #LR = 0.001
    WORKERS = 12
    MIN_MAX=[800, 1333]
    LR_SCHEDULER = "step"
    SCH_PARAMS = [100, 0.1]

    # loops params
    idx = 0
    lr_list = [0.00005]

    for i in lr_list: 
        # setting up list of models
        conf_list = [configs.conf_maker(TRAIN,
                                        TEST,
                                        "Mask_RCNN_R50_FPN",
                                        "R50_AdamW_"+str(i),
                                        BATCH_SIZE=1,
                                        WORKERS=WORKERS,
                                        LR=i, 
                                        NUM_EPOCHS=EPOCHS,
                                        LOAD_FLAG=LOAD, 
                                        LOAD_BEST=BEST,
                                        MIN_MAX=MIN_MAX, 
                                        OPTIMIZER = "AdamW",
                                        TRANSFORMS="combine_transforms",
                                        LR_SCHEDULER=LR_SCHEDULER,
                                        SCHEDULER_PARAMS=SCH_PARAMS)#,
                     #configs.conf_maker(TRAIN,
                     #                   TEST,
                     #                   "Mask_RCNN_R18_FPN",
                     #                   "R18_step_"+str(i),
                     #                   BATCH_SIZE=BATCH_SIZE,
                     #                   WORKERS=WORKERS,
                     #                   LR=i, 
                     #                   NUM_EPOCHS=EPOCHS,
                     #                   LOAD_FLAG=LOAD, 
                     #                   LOAD_BEST=BEST,
                     #                   MIN_MAX=MIN_MAX,
                     #                   OPTIMIZER = "SGD", 
                     #                   TRANSFORMS="combine_transforms",
                     #                   LR_SCHEDULER=LR_SCHEDULER,
                     #                   SCHEDULER_PARAMS=SCH_PARAMS),
                     #configs.conf_maker(TRAIN,
                     #                   TEST,
                     #                   "Mask_RCNN_R34_FPN",
                     #                   "R34_step_"+str(i),
                     #                   BATCH_SIZE=BATCH_SIZE,
                     #                   WORKERS=WORKERS,
                     #                   LR=i, 
                     #                   NUM_EPOCHS=EPOCHS,
                     #                   LOAD_FLAG=LOAD, 
                     #                   LOAD_BEST=BEST, 
                     #                   MIN_MAX=MIN_MAX,
                     #                   OPTIMIZER = "SGD",
                     #                   TRANSFORMS="combine_transforms",
                     #                   LR_SCHEDULER=LR_SCHEDULER,
                     #                   SCHEDULER_PARAMS=SCH_PARAMS)
                                     ]

        # loop to train models through experiment
        for conf in conf_list:
            # calling main    
            TrainNetwork(conf)
        
        idx += 1

# execution
if __name__ == "__main__":
    main()