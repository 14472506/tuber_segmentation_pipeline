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
    
    plt.figure()
    # plotting data
    plt.plot(epochs, train_loss, label = "training_loss")
    plt.plot(epochs, val_loss, label = "validation_loss")
    
    # labelling and formating plot
    plt.xlabel("number of epochs")
    plt.ylabel("total loss value")
    plt.title(title_string)
    plt.legend()

    # saving plot
    plot_string = save_loc + "/" + title_string + "_lr_loss.png"
    print(plot_string)
    plt.savefig(plot_string)
    plt.close()

def plot_precision_recall(results_dict, title_string, save_loc):

    plt.figure()
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
    plot_string = save_loc + "/" + title_string + "_precision_recall.png"
    plt.savefig(plot_string)    
    
    plt.close()

def plot_f1_score(results_dict, title_string, save_loc):

    plt.figure()
    
    prec_list = results_dict['precision']
    rec_list = results_dict['recall']
    threshold = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    f1_list = []

    for i in range(len(prec_list)):

        f1 = 2*((prec_list[i] * rec_list[i])/(prec_list[i] + rec_list[i]))
        f1_list.append(f1)
    
    plt.plot(threshold, f1_list, label = "f1 score")

    # labelling and formatting plot
    plt.xlabel("recall")
    plt.ylabel("F1 score")
    plt.title(title_string)
    plt.legend()   

    # saving plot
    plot_string = save_loc + "/" + title_string + "_F1_score.png"
    plt.savefig(plot_string)    
    
    plt.close()
    #processing dict