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
experiment_id = "K21NmXO6SXGJwMov2wDxOA"
experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
df = experiment.get_scalars()
print(df["run"].unique())
print(df["tag"].unique())
mins = {}
for exp in df["run"].unique():
    model1 = df[df.run.str.contains(exp)]
    data = model1[model1.tag.str.contains('rms')]
    section = int(len(data['value'].values)/8)
    mins[exp] = np.sum(data['value'].values[-section:])/section
print(mins)
print(mins.keys())
print(mins.values())
"""    
model1  = df[df.run.str.contains('pose')]
#get one line from tensorboard plot
data1 = model1[model1.tag.str.contains('rms normalised')]
# get one plot from tensorboard plot
model2  = df[df.run.str.contains('mike')]
#get one line from tensorboard plot
data2 = model2[model2.tag.str.contains('rms normalised')]

# get x and y data
plt.plot(data1['step'], data1['value'], label='pose invariant semi-supervised')
plt.plot(data2['step'], data2['value'], label='fully-supervised')
plt.legend()
plt.ylabel('RMSE')
plt.xlabel('epochs')
#plt.show()
plt.savefig('two_encoders.png')
"""
