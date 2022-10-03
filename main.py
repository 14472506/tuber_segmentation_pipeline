"""
Detials
"""
# =============================================================================================== #
# Imports
# =============================================================================================== #
from loops import TrainLoop, EvalLoop
import yaml

# =============================================================================================== #
# Main
# =============================================================================================== #
def main():
    # defining list of experiments    
    exp_list = ["config/baseline_r50_adam.yaml",
                "config/ss_pt_r50_adam.yaml",
                "config/baseline_r50_adam2.yaml",
                "config/ss_pt_r50_adam_2.yaml",
                "config/baseline_r50_adam3.yaml",
                "config/ss_pt_r50_adam_3.yaml",
                "config/baseline_r50_adam4.yaml",
                "config/ss_pt_r50_adam_4.yaml"]#,
                #"config/baseline_r50_adam5.yaml",
                #"config/ss_pt_r50_adam_5.yaml"
                #]

    # looping through experiments list calling loop_train
    for exp in exp_list:

        print("running experiment: ", exp)

        with open(exp, "r") as data:
            try: 
                config_dict = yaml.safe_load(data)
            except yaml.YAMLError as exc:
                print(exc)
    
        #TrainLoop(config_dict = config_dict) 
        EvalLoop(config_dict = config_dict)

# =============================================================================================== #
# Execution
# =============================================================================================== #
if __name__ == "__main__":
    main()