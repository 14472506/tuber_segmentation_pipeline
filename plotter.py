"""
Details
"""
# ===========================
# Import libraries/packages
# ===========================
from matplotlib import pyplot as plt

# ===========================
# Functions
# ===========================
def plot_lr_loss(epochs, train_loss, val_loss, title_string, save_loc):
    
    # plotting data
    plt.plot(epochs, train_loss, label = "training_loss")
    plt.plot(epochs, validation_loss, label = "validation_loss")
    
    # labelling and formating plot
    plt.xlabel("number of epochs")
    plt.ylabel("total loss value")
    plt.title(title_string)
    plt.legend()

    # saving plot
    plot_string = save_loc + "/" + title_string
    plt.savefig


