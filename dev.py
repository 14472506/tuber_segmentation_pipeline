x = [1, 2, 3]
y = [2, 4, 6]
ep = [1, 2, 3]
title = "test"
location = "outputs"

from re import T
from plotter import plot_lr_loss

plot_lr_loss(ep,x,y,title,location)