"""
Details
"""
from loop.train_net import TrainNetwork
import config.configs as configs

if __name__ == "__main__":
    
    # conf dict list for experimental setup
    # Default setting as seen below:
    # ==============================
    # conf_maker(TRAIN, TEST, MODEL, OUT_DIR, TRANSFORMS="", LOAD_FLAG=False, LOAD_BEST=True, BATCH_SIZE=2,
    #            WORKERS=4 , MIN_MAX=[800, 1333], LR=0.005, NUM_EPOCHS=20,
    #            TEST_IM_STR="data/jersey_royal_dataset/test/169.JPG"):
    ###############################################################################################
    TRAIN = True
    TEST = False
    LOAD = False
    BEST = True
    
    EPOCHS = 10
    BATCH_SIZE = 1
    WORKERS = 0
    LR_SCHEDULER = "lr_step"
    ###############################################################################################
    
    idx = 1
    lr_list = [0.001]
    
    for i in lr_list:
        
        # setting up list of models

        conf_list = [configs.conf_maker(TRAIN, TEST, "test_selector", "test_"+str(idx), BATCH_SIZE=1,
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