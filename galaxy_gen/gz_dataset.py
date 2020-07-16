from pathlib import Path

import pandas as pd
from PIL import Image
import torch


class GZDataset(torch.utils.data.Dataset):
    def __init__(self, gz_root, transform=lambda x: x):
        """
        gz_root = path to root of the galaxy zoo 2 dataset
        transform = torchvision transform to apply to the loaded images
        use_weighted = whether to use the weighted or unweighted vote counts
        """

        self.root = Path(gz_root)
        df_path = self.root / "gz2_classifications_and_subjects.csv"
        task_name_path = self.root / "gz2_classification_task_names.txt"
        answer_name_path = self.root / "gz2_classification_answer_names.txt"
        self.df = pd.read_csv(df_path.as_posix())
        self.transform = transform
        self.task_names = [line.strip() for line in task_name_path.open()]
        self.answer_names = [line.strip() for line in answer_name_path.open()]

    def __getitem__(self, i):
        entry = self.df.loc[i]
        png_loc = entry["png_loc"]
        jpg_loc = png_loc.replace(".png", ".jpg")
        im = Image.open(self.root / jpg_loc)
        x = self.transform(im)

        answers = [
            list(
                self.df.loc[i][
                    [k for k in self.df.keys() if k.startswith(tn) and "count" in k]
                ]
            )
            for tn in self.task_names
        ]

        # this is pretty disgusting, but it's the only way I could think of to get this to
        # work with pytorch's dataloader assumptions.
        return (
            x,
            torch.tensor(answers[0]),
            torch.tensor(answers[1]),
            torch.tensor(answers[2]),
            torch.tensor(answers[3]),
            torch.tensor(answers[4]),
            torch.tensor(answers[5]),
            torch.tensor(answers[6]),
            torch.tensor(answers[7]),
            torch.tensor(answers[8]),
            torch.tensor(answers[9]),
            torch.tensor(answers[10]),
        )

    def __len__(self):
        return len(self.df["total_classifications"])
