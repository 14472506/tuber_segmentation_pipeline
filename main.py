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
    exp_list = ["dev.yaml"]

    # looping through experiments list calling loop_train
    for exp in exp_list:

        print("running experiment: ", exp)

        with open(exp, "r") as data:
            try: 
                config_dict = yaml.safe_load(data)
            except yaml.YAMLError as exc:
                print(exc)
    
        TrainLoop(config_dict = config_dict) 
        EvalLoop(config_dict = config_dict)

# =============================================================================================== #
# Execution
# =============================================================================================== #
if __name__ == "__main__":
    main()