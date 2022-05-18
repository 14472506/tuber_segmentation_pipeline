"""
Details
"""
def conf_maker(TRAIN,
               TEST,
               MODEL,  
               OUT_DIR,
               TRANSFORMS="",
               LOAD_FLAG=False,
               LOAD_BEST=True,
               BATCH_SIZE=2,
               WORKERS=4,             
               MIN_MAX=[800, 1333], 
               LR=0.005,
               NUM_EPOCHS=20,
               TEST_IM_STR="data/jersey_royal_dataset/test/169.JPG"):
    """
    conf_maker generates a dictionary that is passed through the training/testing process 
    """
    # initialsi conf_dict
    config_dict = {}

    # data set locations
    config_dict['train_dir'] = "data/jersey_royal_dataset/train"
    config_dict['train_json'] = "data/jersey_royal_dataset/train/train.json"
    config_dict['validate_dir'] = "data/jersey_royal_dataset/val"
    config_dict['validate_json'] = "data/jersey_royal_dataset/val/val.json"
    config_dict['test_dir'] = "data/jersey_royal_dataset/test"
    config_dict['test_json'] = "data/jersey_royal_dataset/test/test.json"

    # Train and Test flags
    config_dict['TRAIN'] = TRAIN
    config_dict['TEST'] = TEST

    # Transform configs
    config_dict['transforms'] = TRANSFORMS

    # Dataloader config
    config_dict['batch_size'] = BATCH_SIZE
    config_dict['loader_shuffle'] = True
    config_dict['loader_workers'] = WORKERS

    # Model config
    config_dict['model'] = MODEL
    config_dict['num_classes'] = 2
    config_dict['min_max'] = MIN_MAX

    # optimizer config
    config_dict['optimizer'] = "SGD"
    config_dict['optimizer_params'] = {'lr': LR,
                                       'momentum': 0.9,
                                       'weight_decay': 0.0005
                                      }

    # lr_scheduler
    config_dict['lr_scheduler'] = "" 

    # training loop config
    config_dict['num_epochs'] = NUM_EPOCHS
    config_dict['print_freq'] = 10

    # saving and load config
    config_dict['out_dir'] = "outputs/" + OUT_DIR
    config_dict['load_flag'] = LOAD_FLAG
    
    
    if LOAD_BEST:
        LOAD_MODEL = "best_model"
    else:
        LOAD_MODEL = "last_model" 
        
    config_dict['load'] = "outputs/" + OUT_DIR + "/" + LOAD_MODEL + ".pth"
    
    # plotting detials
    config_dict['plot_title'] = OUT_DIR + "_" + LOAD_MODEL + "_lr: " + str (LR)
    config_dict['im_test_path'] = TEST_IM_STR
    
    return config_dict