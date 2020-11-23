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
experiment_id = "2YxbYeGXTyKuD0nYCOALbA"
plot_name = "data_redun_rms.png"
experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
plot_label = "RMSE"
contains = "rms"

df = experiment.get_scalars()
print(df["run"].unique())
print(df["tag"].unique())

for exp in df["run"].unique():
    model1  = df[df.run.str.contains(exp)]
    data = model1[model1.tag.str.contains(contains)] 
    if exp == "0.9":
        plt.plot(data['step'].to_numpy()[0:200], data['value'].to_numpy()[0:200], label=exp)
    else:
        plt.plot(data['step'].to_numpy(), data['value'].to_numpy(), label=exp)
plt.legend()
plt.ylabel(plot_label)
plt.xlabel('epochs')
plt.savefig(plot_name)


    
