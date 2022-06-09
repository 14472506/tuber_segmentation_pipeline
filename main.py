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
    EPOCHS = 10
    BATCH_SIZE = 2
    WORKERS = 0
    LR_SCHEDULER = "lr_step"
    SCH_PARAMS = [2, 0.001]

    # loops params
    idx = 3
    lr_list = [0.005]

    for i in lr_list: 
        # setting up list of models
        conf_list = [configs.conf_maker(TRAIN, TEST, "Mask_RCNN_R50_FPN", "test_"+str(idx), BATCH_SIZE=BATCH_SIZE,
                                        WORKERS=WORKERS, LR=i, NUM_EPOCHS=EPOCHS, LOAD_FLAG=LOAD, LOAD_BEST=BEST, 
                                        TRANSFORMS="combine_transforms", LR_SCHEDULER=LR_SCHEDULER, SCHEDULER_PARAMS=SCH_PARAMS)#,
                    #configs.conf_maker(TRAIN, TEST, "Mask_RCNN_R50_FPN", "Colour_T_"+str(idx), BATCH_SIZE=1,
                    #                    WORKERS=WORKERS, LR=i, NUM_EPOCHS=EPOCHS, LOAD_FLAG=LOAD, LOAD_BEST=BEST, 
                    #                    TRANSFORMS="colour_transforms"),
                    #configs.conf_maker(TRAIN, TEST, "Mask_RCNN_R50_FPN", "Combine_T_"+str(idx), BATCH_SIZE=1,
                    #                    WORKERS=WORKERS, LR=i, NUM_EPOCHS=EPOCHS, LOAD_FLAG=LOAD, LOAD_BEST=BEST, 
                    #                    TRANSFORMS="combine_transforms")
                    ]

        # loop to train models through experiment
        for conf in conf_list:
            # calling main    
            TrainNetwork(conf)
        
        idx += 1

# execution
if __name__ == "__main__":
    main()