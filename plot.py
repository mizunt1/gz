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

experiment_id = "HR0KIPDWSW2OuY1CK08nng"
plot_name = "two_step_1200_test.png"
experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
train = True
plot_what = "Test loss classifier"
contains = plot_what


df = experiment.get_scalars()
print(df["run"].unique())
print(df["tag"].unique())
model1  = df[df.run.str.contains('wod')]
#get one line from tensorboard plot
data1 = model1[model1.tag.str.contains(contains)]
# get one plot from tensorboard plot
model2  = df[df.run.str.contains('ss')]
#get one line from tensorboard plot
data_ss = model2[model2.tag.str.contains(contains)]
model3  = df[df.run.str.contains('2step')]
#get one line from tensorboard plot
data_2s = model3[model3.tag.str.contains(contains)]





# get x and y data

fig, ax1 = plt.subplots()

color = 'tab:orange'
ax1.set_xlabel('semi-supervised steps')

ax1.set_ylabel(plot_what)
ln1 = ax1.plot(data_ss['step'], data_ss['value'], label='semi-supervised with pose', color=color)
ax1.tick_params(axis='x')

ax2 = ax1.twiny()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'

ax2.set_xlabel('fully-supervised steps')
ln2 = ax2.plot((data1['step']), data1['value'], label='fully-supervised with pose', color=color)
color = 'tab:green'
ln2 = ax2.plot((data_2s['step']), data_2s['value'], label='fully-supervised with pre-trained VAE on unlabelled data', color=color)
ax2.tick_params(axis='x')

fig.tight_layout()  # otherwise the right y-label is slightly clipped
#ax1.legend()
#ax2.legend()
#plt.legend(handles=[ax1, ax2])

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc=1)


plt.savefig(plot_name)
