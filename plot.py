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

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb
experiment_id = "2JqRypetRCGPdwtaRNmK5A"
experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
df = experiment.get_scalars()
print(df["run"].unique())
print(df["tag"].unique())

model1  = df[df.run.str.contains('wod_mike')]
#get one line from tensorboard plot
data1 = model1[model1.tag.str.contains('rms normalised')]
# get one plot from tensorboard plot
model2  = df[df.run.str.contains('nr3')]
#get one line from tensorboard plot
data2 = model2[model2.tag.str.contains('rms normalised')]

# get x and y data
plt.plot(data1['step'], data1['value'], label='wod_mike')
plt.plot(data2['step'], data2['value'], label='rms_normalised')
plt.legend()
plt.ylabel('rmse')
plt.xlabel('epochs')
plt.show()
