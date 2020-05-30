import pandas as pd
import torch
import os
from skimage import io

class Gz2_data(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, list_of_interest):
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.file_path = os.path.join(self.root_dir, self.csv_file)
        self.file = pd.read_csv(self.file_path)
        self.root_dir = root_dir
        self.list_of_interest = list_of_interest

    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir, self.file['png_loc'][idx])
        img_name = img_name.replace(".png", ".jpg")
        image = io.imread(img_name)
        data = self.file.iloc[idx][self.list_of_interest]
        data = torch.tensor(data.values.astype('int32'))
        image = torch.tensor(image)
        # permute to NCHW
        image = image.permute(2,0,1).float()
        sample = {'image': image, 'data': data}

        return sample
    
if __name__ == "__main__":
    a01 = "t01_smooth_or_features_a01_smooth_count"
    a02 = "t01_smooth_or_features_a02_features_or_disk_count"
    a03 = "t01_smooth_or_features_a03_star_or_artifact_count"
    data = Gz2_data(csv_file="gz2_20.csv",
                    root_dir="~/diss/gz2_data",
                    list_of_interest=[a01,
                                      a02,
                    a03])
    sample_of_data = data[1]
    print(type(sample_of_data['data']))
    print(type(sample_of_data['image']))
