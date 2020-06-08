import pandas as pd
import torch
import os
from PIL import Image
from skimage import io
import torchvision as tv

class Gz2_data(torch.utils.data.Dataset):
    def __init__(self, csv_dir, image_dir, list_of_interest, faulty_data_set=False):
        self.csv_dir = csv_dir
        self.faulty_data_set = faulty_data_set
        self.image_dir = image_dir
        self.file = pd.read_csv(self.csv_dir)
        self.list_of_interest = list_of_interest

    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.faulty_data_set == True:
            image = None
            while image is None:
                try:
                    img_name = os.path.join(self.image_dir, self.file['png_loc'][idx])
                    img_name = img_name.replace(".png", ".jpg")
                    image = Image.open(img_name).convert('L')
                except FileNotFoundError:
                    idx += 1
        else:
            img_name = os.path.join(self.image_dir, self.file['png_loc'][idx])
            img_name = img_name.replace(".png", ".jpg")
            image = Image.open(img_name).convert('L')

        data = self.file.iloc[idx][self.list_of_interest]
        data = torch.tensor(data.values.astype('int32'))
        trans = tv.transforms.ToTensor()
        image = trans(image)
        print(image.shape)
        sample = {'image': image, 'data': data}

        return sample
    
if __name__ == "__main__":
    a01 = "t01_smooth_or_features_a01_smooth_count"
    a02 = "t01_smooth_or_features_a02_features_or_disk_count"
    a03 = "t01_smooth_or_features_a03_star_or_artifact_count"
    # not sure why PIL doesnt work with ~
    data = Gz2_data(csv_dir="~/diss/gz2_data/gz_amended.csv",
                    image_dir="/Users/Mizunt/diss/gz2_data",
                    list_of_interest=[a01,
                                      a02,
                                      a03], faulty_data_set=False)
    sample_of_data = data[1]
    print(type(sample_of_data['data']))
    print(sample_of_data['image'].shape)
    print(type(sample_of_data['image']))
