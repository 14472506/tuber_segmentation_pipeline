"""
Details
"""
# imports
import torch

from models import model_selector
from .data_loader import PipelineDataLoader

# classes
class TestNet():
    """
    Detials
    """
    def __init__(self, configuration_dict):
        """
        Detials
        """
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.seed = seed
        self.out_dir = configuration_dict["out_dir"]
        self.test = configuration_dict['TEST']
        self.train = False
        self.transforms_list = ""
        self.batch_size = configuration_dict["batch_size"]
        self.loader_shuffle = configuration_dict["loader_shuffle"]
        self.loader_workers = configuration_dict["loader_workers"]
        self.train_dir = configuration_dict["test_dir"]
        self.train_json = configuration_dict["test_json"]

        # loading model
        self.model_name = configuration_dict["model"]
        self.num_classes = configuration_dict["num_classes"]
        self.max_min = configuration_dict["max_min"]
        self.model = model_selector(self.model_name, self.num_classes, 
                                    self.max_min)

        # load weights
        self.load_checkpoint = configuration_dict["load"]
        checkpoint = torch.load(self.load_checkpoint)
        self.model.load_state_dict(checkpoint["state_dict"])

        self.test_data = {
            "mAP_results": None,
            "Centroid_results": None,
            "fps_results": None
        }

        # getting data loader
        self.loader_configuration_dict = {
            "seed": self.seed,
            "transforms": self.transforms, 
            "train": self.train,
            "test": self.test,
            "batch_size": self.batch_size, 
            "loader_shuffle": self.loader_shuffle,
            "loader_workers": self.loader_workers,
            "test_dir" : self.test_dir,
            "test_json" : self.test_json,
   
        }
        self.loader_assigner(self.loader_configuration_dict)

        # getting mAP
        self.mAP_eval()

    def loader_assigner(self, conf_dict):
        """
        detials
        """
        data_loader = PipelineDataLoader(conf_dict)
        tr_load, v_load, _ = data_loader.manager()
        self.train_loader = tr_load
        self.val_loader = v_load

    def mAP_eval(self):
        """
        details
        """
        evaluate(self.model, self.test_loader, self.device, self.out_dir, test_flag=self.test)

    def fps_evaluate(model, image_path, device):
        """
        details
        Args:
            model (_type_): _description_
            image_path (_type_): _description_
            device (_type_): _description_
        """
        # loading image
        #model.eval()
        img = Image.open(image_path)
        transform = T.Compose([T.ToTensor()])
        img = transform(img)
        img = img.to(device)

        # init times list
        times = []
        model.eval()
        with torch.no_grad():
          for i in range(10):
              start_time = time.time()

              model.eval()
              with torch.no_grad():
                  pred = model([img])

              delta = time.time() - start_time
              times.append(delta)
        mean_delta = np.array(times).mean()
        fps = 1 / mean_delta
        return(fps)