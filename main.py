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
    BATCH_SIZE = 1
    WORKERS = 0
    LR_SCHEDULER = "lr_step"

    # loops params
    idx = 1
    lr_list = [0.001]

    for i in lr_list: 
        # setting up list of models
        conf_list = [configs.conf_maker(TRAIN, TEST, "Mask_RCNN_R50_FPN", "test_"+str(idx), BATCH_SIZE=1,
                                        WORKERS=WORKERS, LR=i, NUM_EPOCHS=EPOCHS, LOAD_FLAG=LOAD, LOAD_BEST=BEST, 
                                        TRANSFORMS="combine_transforms")#,
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