"""
Detials
"""
# imports
# -- non needed so far
# config dict maker class
class ConfMaker():
    """
    Detials
    """

    def __init__(self,
                 dataset_name,
                 train = True,
                 test = True,
                 ):
        """
        Detials
        """
        # object init_params
        self.dataset_name = dataset_name
        self.train = train
        self.test = test
        
        # object dict
        self.dict = {}
    
    def dataset_seletor(self):
        """
        Detials
        """
        # defining root, train, test, val, and json
        root = "data/" + self.dataset_name
        train = "/train"
        test = "/test"
        val = "/val"
        json = ".json"

        # adding dataset detials to dict
        self.dict['train_dir'] = root + train
        self.dict['train_json'] = root + train + "/" + train + json
        self.dict['val_dir'] = root + val
        self.dict['val_json'] = root + val + "/" + val + json
        self.dict['test_dir'] = root + test
        self.dict['test_json'] = root + test + "/" + test + json
    
    def train_test(self):
        """
        Detials
        """
        self.dict['TRAIN'] = self.train
        self.dict['TEST'] = self.test
    
    def dataloader_config(self, 
                          transforms = "None",
                          batch_size = 1,
                          workers = 4, 
                          shuffle = True):
        """
        Detials
        """
        # Transform configs
        self.dict['transforms'] = transforms

        # Dataloader config
        self.dict['batch_size'] = batch_size
        self.dict['loader_shuffle'] = shuffle
        self.dict['loader_workers'] = workers
    
    def model_config(self, 
                     model = "Mask_RCNN_R50_FPN",
                     num_classes = 2,
                     min_max = [800, 1333]):
        """
        Detials
        """
        self.dict['model'] = model
        self.dict['num_classes'] = num_classes
        self.dict['min_max'] = min_max
    
    def optimizer_scheduler_config(self, 
                                   optimizer = "SGD",
                                   lr = 0.005, 
                                   momentum = 0.9,
                                   decay = 0.0005,
                                   scheduler = "step",
                                   scheduler_params = [50, 0.1]):
        """
        Details
        """
        # optimizer config
        self.dict['optimizer'] = optimizer
        if optimizer == "SGD":
            self.dict['optimizer_params'] = {'lr': lr,
                                             'momentum': momentum,
                                             'weight_decay': decay
                                            }
        if optimizer == "Adam":
            self.dict['optimizer_params'] = {'lr': lr
                                            }

        # lr_scheduler
        self.dict['lr_scheduler'] = scheduler
        self.dict['scheduler_params'] = scheduler_params
    
    def loop_config(self, epochs = 100, print_freque = 10):
        """
        Details
        """
        # training loop config
        self.dict['num_epochs'] = epochs
        self.dict['print_freq'] = print_freque
    
    def save_and_load_config(self,
                             output_dir = "test_run",
                             load_model = False,
                             best_model = True):
        """
        Details
        """
        # setting output and load marker
        self.dict['out_dir'] = "outputs/" + output_dir
        self.dict['load_flag'] = load_model
        
        # selecting which model to load if so
        if best_model:
            model = "best_model"
        else:
            model = "last_model"   
        
        # constructing load string
        self.dict['load'] = "outputs/" + output_dir + "/" + model + ".pth"
    
    def get_config(self):
        return self.dict

    def config_test(self):
        self.dataset_seletor()
        self.train_test()
        self.dataloader_config()
        self.model_config()
        self.optimizer_scheduler_config()
        self.loop_config()
        self.save_and_load_config()
        dict = self.get_config()
        return(dict)


