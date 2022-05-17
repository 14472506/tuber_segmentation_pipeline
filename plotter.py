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
def plot_lr_loss(loss_dict, title_string, save_loc):
    
    # getting epoch list
    num_epochs = len(loss_dict['train_loss']['total']) + 1
    epochs = list(range(1, num_epochs))
    
    plt.figure()
    # plotting trainin data
    plt.plot(epochs, loss_dict['train_loss']['total'], label = "training_total", color = '#0000FF')
    plt.plot(epochs, loss_dict['train_loss']['classifier'], label = "training_classifier", color = '#088F8F')
    plt.plot(epochs, loss_dict['train_loss']['box_reg'], label = "training_box_reg", color = '#7393B3')
    plt.plot(epochs, loss_dict['train_loss']['mask'], label = "training_mask", color = '#5F9EA0')
    plt.plot(epochs, loss_dict['train_loss']['objectness'], label = "training_objectness", color = '#0096FF')
    plt.plot(epochs, loss_dict['train_loss']['rpn_box_reg'], label = "training_rpn_box_reg", color = '#00FFFF')
    
    # plotting val dataa
    plt.plot(epochs, loss_dict['val_loss']['total'], label = "val_total", color = '#EE4B2B')
    plt.plot(epochs, loss_dict['val_loss']['classifier'], label = "val_classifier", color = '#880808')
    plt.plot(epochs, loss_dict['val_loss']['box_reg'], label = "val_box_reg", color = '#AA4A44')
    plt.plot(epochs, loss_dict['val_loss']['mask'], label = "val_mask", color = '#800020')
    plt.plot(epochs, loss_dict['val_loss']['objectness'], label = "val_objectness", color = '#CC5500')
    plt.plot(epochs, loss_dict['val_loss']['rpn_box_reg'], label = "val_rpn_box_reg", color = '#E97451')
    
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