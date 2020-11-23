###
# put your directory with tensorboard event files online using
# tensorboard dev upload --logdir <logdir>.
# then get the experiment id from there and put it in "experiment id"
# this logdir contains two directories, for two experiments 
# this example includes taking two lines (two different experiments)
# on the graph "rms normalised" graph on tensorboard
# these two lines are called "wod_mike" and "nr3"
# the rmse for these experiments are then plotted using matplotlib
###

from packaging import version
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb
experiment_id = "CBwSJ4XoTLCNGIbdH6skvw"
plot_name = "detach_RMSE.png"
experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
plot_label = "RMSE"
contains = "rms"

df = experiment.get_scalars()
print(df["run"].unique())
print(df["tag"].unique())
model1  = df[df.run.str.contains('detach')]
#get one line from tensorboard plot
data1 = model1[df.tag.str.contains(contains)]
# get one plot from tensorboard plot
model2  = df[df.run.str.contains('trainer')]
#get one line from tensorboard plot
data2 = model2[model2.tag.str.contains(contains)]
max_num_step = np.max(data1['step'])

print(max_num_step)

plt.plot(data1['step'], data1['value'], label='classifier detached')
plt.plot(data2['step'][0:max_num_step], data2['value'][0:max_num_step], label='classifier and encoder connected')

plt.legend()
plt.ylabel(plot_label)
plt.xlabel('epochs')
plt.savefig(plot_name)


    
