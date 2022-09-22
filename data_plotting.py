import json
import os
from matplotlib import pyplot as plt
from collections import OrderedDict

results_list = []
root = "outputs/"

for root, dir, files in os.walk(root):
    for dirname in sorted(dir):
        f_root = root + dirname + "/" + "training_data.json"
        with open(f_root, "rb") as f:
            data = json.load(f)      
        results_list.append((dirname, data))
 
results_dicts = OrderedDict(results_list)

figure = plt.figure(figsize=(1,2))
rows = 1
columns = 2
h = 50
w = 50
count = 1
for key, val in results_dicts.items():
    
    epochs = list(range(1, len(val["train_total"])+1))
    
    ax = figure.add_subplot(rows, columns, count)
    count += 1
    ax.plot(epochs, val["train_total"], label = "training_total", color = "#0000FF")
    ax.plot(epochs, val["val_total"], label = "val_total", color = "#EE4B2B")
    ax.scatter(val["best_epoch"][-1], val["best_mAP"][-1], c = "g", marker = "o")
    ax.scatter(val["step_epoch"][0], val["step_mAP"][0], c = "g", marker = "o")

    ax.set_xlabel("Number of Epochs")
    ax.set_ylabel("Loss Value")

    plt.title(key)
    best_label = str(round(val["best_mAP"][-1], 3)) + " @ " + str(val["best_epoch"][-1])
    plt.annotate(best_label, (val["best_epoch"][-1], val["best_mAP"][-1]))
    step_label = str(round(val["step_mAP"][0], 3)) + " @ " + str(val["step_epoch"][0])
    plt.annotate(step_label, (val["step_epoch"][0], val["step_mAP"][0]))

#figure.legend(loc="upper right")
plt.show()