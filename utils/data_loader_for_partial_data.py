import pandas as pd
import torch
import os
from skimage import io

class Gz2_data(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, list_of_interest=None):
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.file_path = os.path.join(self.root_dir, self.csv_file)
        self.file = pd.read_csv(self.file_path)
        self.root_dir = root_dir
        self.list_of_interest = list_of_interest

    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx, falty_data_set=False):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if falty_data_set == True:
            image = None
            while image is None:
                try:
                    img_name = os.path.join(self.root_dir, self.file['png_loc'][idx])
                    img_name = img_name.replace(".png", ".jpg")
                    image = io.imread(img_name)
                except FileNotFoundError:
                    idx += 1
        else:
            img_name = os.path.join(self.root_dir, self.file['png_loc'][idx])
            img_name = img_name.replace(".png", ".jpg")
            image = io.imread(img_name)
        if self.list_of_interest is not None:
            data = self.file.iloc[idx][self.list_of_interest]
        else:
            data = self.file.iloc[idx]
        #data = torch.tensor(data.values.astype('int32'))
        image = torch.tensor(image)
        # permute to NCHW
        image = image.permute(2,0,1).float()
        sample = {'image': image, 'data': data}

        return sample
    
if __name__ == "__main__":
    a01 = "t01_smooth_or_features_a01_smooth_count"
    a02 = "t01_smooth_or_features_a02_features_or_disk_count"
    a03 = "t01_smooth_or_features_a03_star_or_artifact_count"
    print("loading data")
    data = Gz2_data(csv_file="gz2_classifications_and_subjects.csv",
                    root_dir="~/diss/gz2_data")
    print("loaded")
    new_frame = pd.DataFrame(columns=data.file.columns)
    j = 0
    i = 0
    while j in range(1000):
        file_path = "gz2_data/" + data.file['png_loc'][i]
        file_path = file_path.replace(".png", ".jpg")  
        if os.path.exists(file_path):
            new_frame.loc[j] = data.file.loc[i]
            j+=1
        i+=1

        
    new_frame.to_csv("gz_amended.csv")
    
