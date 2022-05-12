"""
Details
"""
# ===========================
# Import libraries/packages
# ===========================
from cProfile import label
from turtle import title
from unittest import result
from matplotlib import pyplot as plt

# ===========================
# Functions
# ===========================
def plot_lr_loss(epochs, train_loss, val_loss, title_string, save_loc):
    
    # plotting data
    plt.plot(epochs, train_loss, label = "training_loss")
    plt.plot(epochs, val_loss, label = "validation_loss")
    
    # labelling and formating plot
    plt.xlabel("number of epochs")
    plt.ylabel("total loss value")
    plt.title(title_string)
    plt.legend()

    # saving plot
    plot_string = save_loc + "/" + title_string + ".png"
    print(plot_string)
    plt.savefig(plot_string)

def plot_precision_recall(results_dict, title_string, save_loc):

    # looping through dict to build plot
    for key, val in results_dict.items():
        if key != 'recall':
            plt.plot(results_dict['recall'], val, label = key)
    
    # labelling and formatting plot
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title(title_string)
    plt.legend()

    # saving plot
    plot_string = save_loc + "/" + title_string + ".png"
    print(plot_string)
    plt.savefig(plot_string)    

def f1_score(result_dict, title_string, save_loc):

    #processing dict


