"""
Details
"""
# ===========================
# Import libraries/packages
# ===========================
from matplotlib import pyplot as plt

from loop.engine import centroid_error

# ===========================
# Functions
# ===========================
class ResultPlotter:
    """
    Detials
    """
    def __init__(self, plot_title, save_loc, prec_rec_dict, cent_error):
        """
        Detials
        """
        self.plot_title = plot_title
        self.save_loc = save_loc
        self.precision_recall_dict = prec_rec_dict
        self.cent_error = cent_error
    
    def plot_prec_rec(self):

        plt.figure()
        for key, val in self.precision_recall_dict.items():
            if key != 'recall':
                plt.plot(self.precision_recall_dict['recall'], val, label=key)
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.title(self.plot_title)
        plt.legend()

        plot_string = self.save_loc + "/" + self.plot_title + "_precision_recall.png"
        plt.savefig(plot_string)    
    
        plt.close()

    def plot_f1_score(self):
        print("something")

    def plot_cent_error(self):
 
        thresh_list = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        avg_cent_err = []
    
        for key ,item in self.cent_error.items():
            list_avgs = []
        
            for list in item:
                # calculating average
                avg = round(sum(list)/len(list), 3)
                list_avgs.append(avg)
        
            avg2 = round(sum(list_avgs)/len(list_avgs), 3)
            avg_cent_err.append(avg2)

def plot_lr_loss(loss_dict, title_string, save_loc):
    
    # getting epoch list
    num_epochs = len(loss_dict['train_loss']['total']) + 1
    epochs = list(range(1, num_epochs))

    # Initialising figure
    fig = plt.figure()

    # creating subtplots for loss plot
    ax = fig.add_subplot()
    ax.plot(epochs, loss_dict['train_loss']['total'], label = "training_total", color = '#0000FF')
    ax.plot(epochs, loss_dict['val_loss']['total'], label = "val_total", color = '#EE4B2B')

    # making y marker
    line_list = ['dashed','dashdot','dotted']
    # plotting best modes for each step
    for i in range(len(loss_dict['step_val']['epoch'])):
        string = "best_step_" + str(i)    
        ax.axvline(x=loss_dict['step_val']['epoch'][i], color='#808080', label=string, linestyle=line_list[i])

    ax.set_xlabel("Number of Epochs")
    ax.set_ylabel("Loss Value")

    # creating subplot for Validation mAP
    ax2 = ax2 = ax.twinx()
    ax2.plot(epochs, loss_dict['val_eval']['mAP'], label = "val_mAP", color = '#00FF00')
    ax2.yaxis.tick_right()
    ax2.set_ylabel("Validation_mAP")

    # plotting trainin data
    #plt.plot(epochs, loss_dict['train_loss']['total'], label = "training_total", color = '#0000FF')
    #plt.plot(epochs, loss_dict['train_loss']['classifier'], label = "training_classifier", color = '#088F8F')
    #plt.plot(epochs, loss_dict['train_loss']['box_reg'], label = "training_box_reg", color = '#7393B3')
    #plt.plot(epochs, loss_dict['train_loss']['mask'], label = "training_mask", color = '#5F9EA0')
    #plt.plot(epochs, loss_dict['train_loss']['objectness'], label = "training_objectness", color = '#0096FF')
    #plt.plot(epochs, loss_dict['train_loss']['rpn_box_reg'], label = "training_rpn_box_reg", color = '#00FFFF')

    # plotting val dataa
    #plt.plot(epochs, loss_dict['val_loss']['total'], label = "val_total", color = '#EE4B2B')
    #plt.plot(epochs, loss_dict['val_loss']['classifier'], label = "val_classifier", color = '#880808')
    #plt.plot(epochs, loss_dict['val_loss']['box_reg'], label = "val_box_reg", color = '#AA4A44')
    #plt.plot(epochs, loss_dict['val_loss']['mask'], label = "val_mask", color = '#800020')
    #plt.plot(epochs, loss_dict['val_loss']['objectness'], label = "val_objectness", color = '#CC5500')
    #plt.plot(epochs, loss_dict['val_loss']['rpn_box_reg'], label = "val_rpn_box_reg", color = '#E97451')

    # labelling and formating plot

    plt.title(title_string)
    fig.legend(loc='upper left')

    plt.show()

    # saving plot
    plot_string = save_loc + "/" + title_string + "_lr_loss.png"
    print(plot_string)
    fig.savefig(plot_string)
    plt.close(fig)

###################################################################################################
def plot_f1_score(results_dict, title_string, save_loc):

    plt.figure()
    
    prec_list = results_dict['segm']
    threshold = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    #f1_list = []

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
    
def plot_cent_err(results_dict, title_string, save_loc):
    
    # threshold list 
    thresh_list = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    avg_cent_err = []
    
    # iteratating though results dict
    for key ,item in results_dict.items():
        # list of averages from list
        list_avgs = []
        
        # iterating though lists in item list
        for list in item:
            # calculating average
            avg = round(sum(list)/len(list), 3)
            list_avgs.append(avg)
        
        print(list_avgs)
        avg2 = round(sum(list_avgs)/len(list_avgs), 3)
        avg_cent_err.append(avg2)
            
            
    print(avg_cent_err)
