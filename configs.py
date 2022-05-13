"""
Details
"""

def dev_test():
    # initialsi conf_dict
    config_dict = {}

    # Train and Test flags
    config_dict['TRAIN'] = True
    config_dict['TEST'] = True

    # Transform configs
    config_dict['transforms'] = ""

    # Dataset configs
    config_dict['train_dir'] = "data/jersey_royal_dataset/train"
    config_dict['train_json'] = "data/jersey_royal_dataset/train/train.json"

    config_dict['validate_dir'] = "data/jersey_royal_dataset/val"
    config_dict['validate_json'] = "data/jersey_royal_dataset/val/val.json"

    config_dict['test_dir'] = "data/jersey_royal_dataset/test"
    config_dict['test_json'] = "data/jersey_royal_dataset/test/test.json"

    # Dataloader config
    config_dict['batch_size'] = 2
    config_dict['loader_shuffle'] = True
    config_dict['loader_workers'] = 4

    # Model config
    config_dict['model'] = "Mask_RCNN_R50_FPN"
    config_dict['num_classes'] = 2

    # optimizer config
    config_dict['optimizer'] = "SGD"
    config_dict['optimizer_params'] = {'lr': 0.005,
                                       'momentum': 0.9,
                                       'weight_decay': 0.0005
                                      }

    # lr_scheduler
    config_dict['lr_scheduler'] = "" 

    # training loop config
    config_dict['num_epochs'] = 1
    config_dict['print_freq'] = 10

    # saving and load config
    config_dict['out_dir'] = "outputs/dev_test"
    config_dict['load'] = "" #"outputs/dev_test/last_model.pth"
    
    # plotting detials
    config_dict['plot_title'] = "Mask_RCNN_R50_FP_lr:0.005"
    config_dict['im_test_path'] = "data/jersey_royal_dataset/test/169.JPG"
    
    return config_dict

def Mask_RCNN_R50_FPN_Base():
    # initialsi conf_dict
    config_dict = {}

    # Train and Test flags
    config_dict['TRAIN'] = False
    config_dict['TEST'] = True

    # Transform configs
    config_dict['transforms'] = ""

    # Dataset configs
    config_dict['train_dir'] = "data/jersey_royal_dataset/train"
    config_dict['train_json'] = "data/jersey_royal_dataset/train/train.json"

    config_dict['validate_dir'] = "data/jersey_royal_dataset/val"
    config_dict['validate_json'] = "data/jersey_royal_dataset/val/val.json"

    config_dict['test_dir'] = "data/jersey_royal_dataset/test"
    config_dict['test_json'] = "data/jersey_royal_dataset/test/test.json"

    # Dataloader config
    config_dict['batch_size'] = 2
    config_dict['loader_shuffle'] = True
    config_dict['loader_workers'] = 4

    # Model config
    config_dict['model'] = "Mask_RCNN_R50_FPN"
    config_dict['num_classes'] = 2

    # optimizer config
    config_dict['optimizer'] = "SGD"
    config_dict['optimizer_params'] = {'lr': 0.005,
                                       'momentum': 0.9,
                                       'weight_decay': 0.0005
                                      }

    # lr_scheduler
    config_dict['lr_scheduler'] = "" 

    # training loop config
    config_dict['num_epochs'] = 20
    config_dict['print_freq'] = 10

    # saving and load config
    config_dict['out_dir'] = "outputs/Mask_RCNN_R50_FPN_Base"
    config_dict['load'] = "outputs/Mask_RCNN_R50_FPN_Base/best_model.pth"

    # plotting detials
    config_dict['plot_title'] = "Mask_RCNN_R50_FPN_lr:0.005"
    config_dict['im_test_path'] = "data/jersey_royal_dataset/test/169.JPG"
    
    return config_dict

def Mask_RCNN_Mobilenet2_Base():
    # initialsi conf_dict
    config_dict = {}

    # Train and Test flags
    config_dict['TRAIN'] = False
    config_dict['TEST'] = True

    # Transform configs
    config_dict['transforms'] = ""

    # Dataset configs
    config_dict['train_dir'] = "data/jersey_royal_dataset/train"
    config_dict['train_json'] = "data/jersey_royal_dataset/train/train.json"

    config_dict['validate_dir'] = "data/jersey_royal_dataset/val"
    config_dict['validate_json'] = "data/jersey_royal_dataset/val/val.json"

    config_dict['test_dir'] = "data/jersey_royal_dataset/test"
    config_dict['test_json'] = "data/jersey_royal_dataset/test/test.json"

    # Dataloader config
    config_dict['batch_size'] = 1
    config_dict['loader_shuffle'] = True
    config_dict['loader_workers'] = 4

    # Model config
    config_dict['model'] = "Mask_RCNN_mobilenetv2"
    config_dict['num_classes'] = 2

    # optimizer config
    config_dict['optimizer'] = "SGD"
    config_dict['optimizer_params'] = {'lr': 0.005,
                                       'momentum': 0.9,
                                       'weight_decay': 0.0005
                                      }

    # lr_scheduler
    config_dict['lr_scheduler'] = "" 

    # training loop config
    config_dict['num_epochs'] = 20
    config_dict['print_freq'] = 10

    # saving and load config
    config_dict['out_dir'] = "outputs/Mask_RCNN_Mobilenet2_Base"
    config_dict['load'] = "outputs/Mask_RCNN_Mobilenet2_Base/best_model.pth"

    # plotting detials
    config_dict['plot_title'] = "Mask_RCNN_MobilenetV2_lr:0.005"
    config_dict['im_test_path'] = "data/jersey_royal_dataset/test/169.JPG"
    
    return config_dict