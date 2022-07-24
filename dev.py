
from matplotlib import pyplot as plt
import json

loss_json = "outputs/Saver_test_5e-05/loss_results.json"

with open(loss_json) as pr_file:
    loss_dict = json.load(pr_file)
    pr_file.close()

title_string = "test"

####################################################################

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