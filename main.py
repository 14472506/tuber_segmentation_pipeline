"""
Details
"""
# import
from training_implementation.config import ConfMaker
from training_implementation.train import TrainNet

def exp1():
    config = ConfMaker("jersey_dataset_v3", test=False)
    
    config.dataset_seletor()

    config.train_test()

    config.dataloader_config(transforms = "combined",
                          batch_size = 2,
                          workers = 4, 
                          shuffle = True)

    config.model_config(model = "Mask_RCNN_R50_FPN",
                      num_classes = 2,
                      min_max = [800, 1333])

    config.optimizer_scheduler_config(optimizer = "SGD",
                                     lr = 0.005, 
                                     momentum = 0.9,
                                     decay = 0.0005,
                                     scheduler = "step",
                                     scheduler_params = [50, 0.1])

    config.loop_config(epochs = 50, print_freque = 10)

    config.save_and_load_config(output_dir = "exp1",
                                load_model = False,
                                best_model = True)
    dict = config.get_config()

    return dict


# train model
conf_dict = exp1()
TrainNet(conf_dict)

