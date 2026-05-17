from PIL import Image
import torch
from torch.utils.data import Dataset


class IUXrayDataset(Dataset):
    def __init__(self, dataframe, tokenizer, image_transform=None, max_length=128):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row["image_path"]).convert("RGB")
        if self.image_transform is not None:
            image = self.image_transform(image)

        encoded_text = self.tokenizer(
            row["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = torch.tensor(row["labels"], dtype=torch.float32)

        return {
            "pixel_values": image,
            "input_ids": encoded_text["input_ids"].squeeze(0),
            "attention_mask": encoded_text["attention_mask"].squeeze(0),
            "labels": labels,
        }