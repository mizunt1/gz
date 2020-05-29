from vae import VAE
from load_gz_data import Gz2_data
from simple_classifier import Classifier
import torch

a01 = "t01_smooth_or_features_a01_smooth_count"
a02 = "t01_smooth_or_features_a02_features_or_disk_count"
a03 = "t01_smooth_or_features_a03_star_or_artifact_count"
data = Gz2_data(csv_file="gz2_20.csv",
                root_dir="~/diss/gz2_data",
                list_of_interest=[a01,
                                  a02,
                                  a03])
sample_of_data = data[1]
# image or data

print(sample_of_data['data'])
print(sample_of_data['image'].shape)
