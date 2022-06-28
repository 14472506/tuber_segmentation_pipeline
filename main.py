"""
Details
"""
# imports
from loop.train_net import TrainNetwork
import config.configs as configs

# main function
def main():

    # Boolean flags
    TRAIN = True
    TEST = False
    LOAD = False
    BEST = True
    
    # othere params
    EPOCHS = 50
    BATCH_SIZE = 2
    WORKERS = 4
    LR_SCHEDULER = "step"
    SCH_PARAMS = [50, 0.1]

    # loops params
    idx = 0
    lr_list = [0.005]

    for i in lr_list: 
        # setting up list of models
        conf_list = [configs.conf_maker(TRAIN,
                                        TEST,
                                        "Mask_RCNN_R50_FPN",
                                        "Saver_test_"+str(i),
                                        BATCH_SIZE=BATCH_SIZE,
                                        WORKERS=WORKERS,
                                        LR=i, 
                                        NUM_EPOCHS=EPOCHS,
                                        LOAD_FLAG=LOAD, 
                                        LOAD_BEST=BEST, 
                                        OPTIMIZER = "SGD",
                                        TRANSFORMS="combine_transforms",
                                        LR_SCHEDULER=LR_SCHEDULER,
                                        SCHEDULER_PARAMS=SCH_PARAMS)#,
                     #configs.conf_maker(TRAIN,
                     #                   TEST,
                     #                   "Mask_RCNN_R18_FPN",
                     #                   "R18_S50-01_"+str(i),
                     #                   BATCH_SIZE=BATCH_SIZE,
                     #                   WORKERS=WORKERS,
                     #                   LR=i, 
                     #                   NUM_EPOCHS=EPOCHS,
                     #                   LOAD_FLAG=LOAD, 
                     #                   LOAD_BEST=BEST, 
                     #                   TRANSFORMS="combine_transforms",
                     #                   LR_SCHEDULER=LR_SCHEDULER,
                     #                   SCHEDULER_PARAMS=SCH_PARAMS),
                     #configs.conf_maker(TRAIN,
                     #                   TEST,
                     #                   "Mask_RCNN_R34_FPN",
                     #                   "R34_S50-01_"+str(i),
                     #                   BATCH_SIZE=BATCH_SIZE,
                     #                   WORKERS=WORKERS,
                     #                   LR=i, 
                     #                   NUM_EPOCHS=EPOCHS,
                     #                   LOAD_FLAG=LOAD, 
                     #                   LOAD_BEST=BEST, 
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