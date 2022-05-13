from time import time
from utils import time_converter
import plotter as plt

seconds = 98.681
sec = time_converter(seconds) 
print(sec)

x = [1,2,3]
y = [2,4,6]
e = [1,2,3]

plt.plot_lr_loss(e,x,y,"dave","outputs")